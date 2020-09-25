import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import secretmanager
import alpaca_trade_api as tradeapi
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
import ta
import math

client = secretmanager.SecretManagerServiceClient()

td_api_key_name = client.secret_version_path('<PROJECT_NAME>', '<KEY_NAME>', 1)
td_api_key_response = client.access_secret_version(td_api_key_name)
td_api_key = td_api_key_response.payload.data.decode('UTF-8')

alpaca_base_url = "https://paper-api.alpaca.markets"

alpaca_api_key_name = client.secret_version_path('<PROJECT_NAME>', '<KEY_NAME>', 2)
alpaca_api_key_response = client.access_secret_version(alpaca_api_key_name)
alpaca_api_key = alpaca_api_key_response.payload.data.decode('UTF-8')

alpaca_secret_key_name = client.secret_version_path('<PROJECT_NAME>', '<KEY_NAME>', 2)
alpaca_secret_key_response = client.access_secret_version(alpaca_secret_key_name)
alpaca_secret_key = alpaca_secret_key_response.payload.data.decode('UTF-8')

alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, alpaca_base_url, 'v2')

IEX_API_KEY_name = client.secret_version_path('<PROJECT_NAME>', '<KEY_NAME>', 1)
IEX_API_KEY_response = client.access_secret_version(IEX_API_KEY_name)
IEX_API_KEY = IEX_API_KEY_response.payload.data.decode('UTF-8')

bq_client = bigquery.Client()
storage_client = storage.Client()

dataset_id = 'equity_data'
dataset_ref = bq_client.dataset(dataset_id)

job_config = bigquery.LoadJobConfig()
job_config.source_format = bigquery.SourceFormat.CSV
job_config.autodetect = True
job_config.ignore_unknown_values = True

today = datetime.today().astimezone(pytz.timezone("America/New_York"))
today_fmt = today.strftime('%Y-%m-%d')

np.seterr(all='ignore')

def get_stocks_with_models():
    bucket = storage_client.get_bucket('<BUCKET_NAME>')
    files = bucket.list_blobs()
    stock_symbols = list(set([file.name.split('_')[0] for file in files]))
    
    print('Stocks with models: {}'.format(', '.join(stock_symbols)))
    return stock_symbols

def get_historical_data_df(symbol):
    # Load the historical stock data from BQ
    
    QUERY = (
        'SELECT symbol, date, time, open, high, low, close, volume FROM ( '
        'SELECT symbol, date, time, open, high, low, volume, '
        'LAST_VALUE(close IGNORE NULLS) OVER (PARTITION BY symbol ORDER BY date ASC, time ASC) AS close, '
        'row_number() OVER (PARTITION BY symbol ORDER BY date DESC, time DESC) AS row '
        'FROM `<PROJECT_NAME>.equity_data.daily_quote_data` AS filtered_table '
        'WHERE symbol="' + symbol + '" '
        ') AS stock_table WHERE row <= 51;'
        )
        
    query_job = bq_client.query(QUERY)  # API request
    hist_data = query_job.result().to_dataframe()
    hist_data['open'].fillna(hist_data['close'], inplace=True)
    hist_data['high'].fillna(hist_data['close'], inplace=True)
    hist_data['low'].fillna(hist_data['close'], inplace=True)
    hist_data = hist_data.drop(['symbol'], axis=1)
    
    # add new 'datetime' column
    hist_data['datetime'] = pd.to_datetime(hist_data['date'] + ' ' + hist_data['time'])
    
    # drop old 'date' and 'time' columns
    hist_data = hist_data.drop(['date', 'time'], axis=1)
    
    # change the name of 'datetime' column to 'date'
    hist_data.rename(columns={'datetime': 'date'}, inplace=True)
    
    hist_data = hist_data.dropna()
    
    enough_data = hist_data.shape[0] >= 51
    
    return hist_data, enough_data

def add_indicators(dataset):
    # Take df returned by get_historical_data_df and returns 
    # a single row df with values for all the financial indicators
    # and another with a closing price column.
    
    # reverse order of dataset
    dataset = dataset.iloc[::-1].reset_index(drop=True)
    
    # add features
    dataset['CV_20'] = dataset['close'].rolling(window=20).std()/dataset['close'].rolling(window=20).mean()
    dataset['CV_50'] = dataset['close'].rolling(window=50).std()/dataset['close'].rolling(window=50).mean()
    dataset['TRIX'] = ta.trend.trix(dataset['close'], n=30, fillna=True)      
    dataset['RSI'] = ta.momentum.RSIIndicator(dataset['close'], n=14, fillna=True).rsi()      
    dataset['Williams %R'] = ta.momentum.WilliamsRIndicator(dataset['high'], dataset['low'], dataset['close'], 7, fillna=True).wr()
    dataset['CCI'] = ta.trend.cci(dataset['high'], dataset['low'], dataset['close'], n=20,c=0.015,fillna=True)
    dataset['MACD'] = ta.trend.MACD(dataset['close'], n_slow=26, n_fast=12, n_sign=9,fillna=True).macd()
    dataset['UO'] = ta.momentum.UltimateOscillator(dataset['high'], dataset['low'], dataset['close'], 7, 14, 28, 4.0, 2.0, 1.0, fillna=True).uo()
    dataset['AwesomeOsc'] = ta.momentum.AwesomeOscillatorIndicator(dataset['high'], dataset['low'],5,34,fillna=True).ao()
    dataset['MFI'] = ta.momentum.MFIIndicator(dataset['high'], dataset['low'], dataset['close'], dataset['volume'], n=14, fillna=True).money_flow_index()
    dataset['ROC'] = ta.momentum.roc(dataset['close'],n=12,fillna=True)
    dataset['StochOsc'] = ta.momentum.StochasticOscillator(dataset['high'], dataset['low'], dataset['close'], n=14, d_n=3, fillna=True).stoch()
    dataset['KAMA'] = ta.momentum.kama(dataset['close'],fillna=True)
    dataset['OBV'] = ta.volume.on_balance_volume(dataset['close'], dataset['volume'], fillna=True)
    dataset['ATR'] = ta.volatility.average_true_range(dataset['high'], dataset['low'], dataset['close'], n=14, fillna=True)
    dataset['ADX'] = ta.trend.adx(dataset['high'], dataset['low'], dataset['close'], n=14, fillna=True)
    
    # a single row with values for all the financial indicators.
    todays_feature_data = dataset.iloc[[-1]]
    
    # drop OHLC for the model data
    model_data = todays_feature_data.iloc[:,6:]
    
    return model_data

def get_decision_dict(stocks_with_models):
    # Loop through the array returned by get_stocks_with_models()
    # and open the associated pickle files for each one. Then pass
    # the algorithm data for that stock by calling the 
    # get_historical_data and then add_indicators methods.
    # Return a dictionary with keys of buy, sell, and hold, whose values 
    # are arrays of the appropriate tickers.
    
    decision_dict = {"buy": [], "sell": [], "hold": []}
    
    bucket = storage_client.get_bucket('<BUCKET_NAME>')
    
    for stock in stocks_with_models:
        model_blob = bucket.get_blob('' + stock + '_svm_current.pkl')
        model = pickle.loads(model_blob.download_as_string())
        scaler_blob = bucket.get_blob('' + stock + '_scaler_current.pkl')
        sc = pickle.loads(scaler_blob.download_as_string())
        
        hist_data, enough_data = get_historical_data_df(stock)
        
        if enough_data:
            stock_data = add_indicators(hist_data)
            scaled_data = sc.transform(stock_data)
            predict = model.predict(scaled_data)
            if predict == 1:
                decision_dict["buy"].append(stock)
            elif predict == 0:
                decision_dict["hold"].append(stock)
            elif predict == -1:
                decision_dict["sell"].append(stock)
                
        else:
            print('Not enough data for stock: {}'.format(stock))
    
    print('List of buys: {}'.format(', '.join(decision_dict["buy"])))
    print('List of holds: {}'.format(', '.join(decision_dict["hold"])))
    print('List of sells: {}'.format(', '.join(decision_dict["sell"])))
    return decision_dict

def get_current_portfolio():
    positions = alpaca_api.list_positions()
    
    position_list = [x.symbol for x in positions]
    
    print('Current buy/sell/hold portfolio: {}'.format(', '.join(position_list)))
    return position_list

def sell_stocks(sell_list, position_list):
    alpaca_api.cancel_all_orders()

    if sell_list:
        print('Stocks to sell for buy/sell/hold portfolio: {}'.format(', '.join([x for x in sell_list if x in position_list])))

    for stock in position_list:
        position = alpaca_api.get_position(stock)
        number_of_shares = position.qty
        if stock in sell_list:
            order = alpaca_api.submit_order(
                    symbol=stock,
                    qty=number_of_shares,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            print('Bought {} at {}, selling at {}'.format(stock, position.avg_entry_price, position.current_price))
        
def get_current_price_and_diff(symbol, buying_price=None):
    # Take a stock symbol and a price. Make an API call to get the
    # current price, then return the current price and percentage gain or loss
    # (if buying price is supplied).
    
    url = 'https://cloud.iexapis.com/stable/stock/{}/price'.format(symbol)
    
    params = {
    'token': IEX_API_KEY,
    }
    
    request = requests.get(
        url=url,
        params=params
        ).json()

    if buying_price:
        return request, round((request - float(buying_price)) * 100/float(buying_price), 2)

    return request

def buy_stocks(buy_list, positions_list):
    new_buys_list = [x for x in buy_list if x not in positions_list]
    
    account = alpaca_api.get_account()

    cash_on_hand = float(account.cash)

    portfolio_value = float(account.portfolio_value)

    print('Cash on hand for buy/sell/hold portfolio: {}'.format(cash_on_hand))
    print('Portfolio value: {}'.format(portfolio_value))
    
    print('Buy/sell/hold stocks to buy: {}'.format(', '.join(new_buys_list)))
        
    if new_buys_list:
        position_size = cash_on_hand / len(new_buys_list)
        for stock in new_buys_list:
            current_stock_price = get_current_price_and_diff(stock)
            number_of_shares = math.floor(position_size / current_stock_price)
            if cash_on_hand > position_size and number_of_shares > 0:
                try:
                    order = alpaca_api.submit_order(
                        symbol=stock,
                        qty=number_of_shares,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    cash_on_hand = cash_on_hand - position_size
                except Exception as e:
                    print('Error with stock {}: {}'.format(stock, e))
    
    
def trade_bot(*args):
    # Driver method, connects all the pieces.
    
    # Check if market is open
    market_url = 'https://api.tdameritrade.com/v1/marketdata/EQUITY/hours'
    params = {
        'apikey': td_api_key,
        'date': today_fmt
    }
    
    request = requests.get(
        url=market_url,
        params=params
    ).json()
    
    try:
        if request['equity']['EQ']['isOpen']:
            print('Getting models')
            stocks_with_models = get_stocks_with_models()
            print('Getting buy/sell decisions')
            decision_dict = get_decision_dict(stocks_with_models)
            print('Getting current portfolio')
            positions_list = get_current_portfolio()
            print('Selling stocks')
            sell_stocks(decision_dict['sell'], positions_list)
            print('Buying stocks')
            buy_stocks(decision_dict['buy'], positions_list)
                        
            return 'Success'

    except KeyError as e:
        # Not a weekday
        print(f'Error: {e}')
        
        return 'Market not open'
