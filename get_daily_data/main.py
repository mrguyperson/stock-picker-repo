# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:15:47 2020

@author: Josh
"""

import pytz
import requests
from google.cloud import bigquery, secretmanager, storage
import time
from datetime import datetime
import numpy as np
import math
import pandas as pd
import itertools
import ta

secret_client = secretmanager.SecretManagerServiceClient()
storage_client = storage.Client()
bq_client = bigquery.Client()

td_api_key_name = secret_client.secret_version_path('<PROJECT_NAME>', '<KEY_NAME>', 1)
td_api_key_response = secret_client.access_secret_version(td_api_key_name)
td_api_key = td_api_key_response.payload.data.decode('UTF-8')

IEX_API_KEY_name = secret_client.secret_version_path('<PROJECT_NAME>', '<KEY_NAME>', 1)
IEX_API_KEY_response = secret_client.access_secret_version(IEX_API_KEY_name)
IEX_API_KEY = IEX_API_KEY_response.payload.data.decode('UTF-8')

def daily_equity_quotes(*args):
    # Check if the market was open today. Cloud functions use UTC and I'm in
    # eastern so I convert the timezone
    today = datetime.today().astimezone(pytz.timezone("America/New_York"))
    today_fmt = today.strftime('%Y-%m-%d')
    time_now = datetime.now().astimezone(pytz.timezone("America/New_York")).strftime("%H:%M")

    # Call the td ameritrade hours endpoint for equities to see if it is open
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
            print('Getting tickers')
            
            symbols_url = 'https://cloud.iexapis.com/beta/ref-data/iex/symbols'

            symbols_params = {
                'token': IEX_API_KEY
            }

            request = requests.get(
                url=symbols_url,
                params=symbols_params
            ).json()

            symbols = [x.get('symbol') for x in request if x.get('isEnabled')]

            def chunks(l, n):
                """
                Takes in a list and how long you want
                each chunk to be
                """
                n = max(1, n)
                return (l[i:i+n] for i in range(0, len(l), n))

            symbols_chunked = list(chunks(symbols, 100))

            def quotes_request(stocks, n):
                """
                Makes an api call for a list of stock symbols
                and returns a dataframe
                """

                stocks_param = ','.join(stocks)
        
                list_of_quotes = []
                
                try:
                    url = 'https://cloud.iexapis.com/stable/stock/market/batch/'
            
                    params = {
                    'token': IEX_API_KEY,
                    'symbols': stocks_param,
                    'types': 'intraday-prices',
                    'chartIEXOnly': True,
                    'chartLast': 30,
                    'chartReset': True
                    }
                
                    price_request = requests.get(
                        url=url,
                        params=params
                    ).json()

                    for key in stocks:
                        if key in price_request and len(price_request[key]["intraday-prices"]) > 0:
                            prices = price_request[key]["intraday-prices"]
                            open_list = [x['open'] for x in prices if x['open'] is not None]
                            open_price = float(open_list[0])if len(open_list) > 0 else None
                            close_list = [x['close'] for x in prices if x['close'] is not None]
                            close_price = float(close_list[-1]) if len(close_list) > 0 else None
                            high_price = np.nanmax(np.array([x.get('high') for x in prices], dtype=np.float64))
                            low_price = np.nanmin(np.array([x.get('low') for x in prices], dtype=np.float64))
                            list_of_quotes.append({
                                'symbol': key,
                                'date': today_fmt,
                                'time': time_now,
                                'high': high_price if not math.isnan(high_price) else None,
                                'low': low_price if not math.isnan(low_price) else None,
                                'volume': np.nansum(np.array([x.get('volume') for x in prices], dtype=np.float64)),
                                'number_of_trades': np.nansum(np.array([x.get('numberOfTrades') for x in prices], dtype=np.float64)),
                                'open': open_price,
                                'close': close_price
                            })
                        else:
                            list_of_quotes.append({
                                'symbol': key,
                                'date': today_fmt,
                                'time': time_now,
                                'high': float('nan'),
                                'low': float('nan'),
                                'volume': float(0),
                                'number_of_trades': float(0),
                                'open': float('nan'),
                                'close': float('nan')
                            })
                        
                except Exception as e:
                    print('Problem with stocks: {}, {}'.format(stocks_param, e))
                    if n >= 2:
                        n = int(n/2)
                        list_of_quotes.extend(list(itertools.chain(*[quotes_request(each, n) for each in list(chunks(stocks, n))])))
                    else:
                        pass

                time.sleep(.5)
                
                return list_of_quotes

            # Loop through the chunked list of synbols
            # and call the api. Append all the resulting dataframes into one
            print('Making quote requests')
            df = pd.concat([pd.DataFrame(quotes_request(each, 100)) for each in symbols_chunked])

            print('Data received: {}'.format(df.head().to_string()))
            print('Number of rows added: {}'.format(len(df.index)))

            dataset_id = 'equity_data'
            table_id = 'daily_quote_data'

            dataset_ref = secret_client.dataset(dataset_id)
            table_ref = dataset_ref.table(table_id)

            print('Adding to bigquery')
            job_config = bigquery.LoadJobConfig()
            job_config.source_format = bigquery.SourceFormat.CSV
            job_config.autodetect = True
            job_config.ignore_unknown_values = True
            job = bq_client.load_table_from_dataframe(
                df,
                table_ref,
                location='US',
                job_config=job_config
            )

            job.result()
            
            print('Dropping old data')
            QUERY = (
                'DELETE FROM `<PROJECT NAME>.equity_data.daily_quote_data` '
                'WHERE CAST(date as date) < DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) '
                'OR date IS NULL'
                )
        
            query_job = bq_client.query(QUERY)  # API request
            print(query_job.result())
            
            def get_stocks_with_models():
                bucket = storage_client.get_bucket('stock_data_bucket_jrth')
                files = bucket.list_blobs()
                stock_symbols = list(set([file.name.split('_')[0] for file in files]))
                
                print('Stocks with models: {}'.format(', '.join(stock_symbols)))
                return stock_symbols
            
            def get_hist_data_and_add_indicators(symbol):
                QUERY = (
                    'SELECT symbol, date, time, open, high, low, close, volume FROM ( '
                    'SELECT symbol, date, time, open, high, low, volume, '
                    'LAST_VALUE(close IGNORE NULLS) OVER (PARTITION BY symbol ORDER BY date ASC, time ASC) AS close, '
                    'row_number() OVER (PARTITION BY symbol ORDER BY date DESC, time DESC) AS row '
                    'FROM `<PROJECT NAME>.equity_data.daily_quote_data` AS filtered_table '
                    'WHERE symbol="' + symbol + '" '
                    ') AS stock_table WHERE row <= 51;'
                    )
                    
                hist_query_job = bq_client.query(QUERY)  # API request
                hist_data = hist_query_job.result().to_dataframe()
                hist_data['open'].fillna(hist_data['close'], inplace=True)
                hist_data['high'].fillna(hist_data['close'], inplace=True)
                hist_data['low'].fillna(hist_data['close'], inplace=True)
                
                # reverse order of dataset
                dataset = hist_data.drop(['symbol'], axis=1).iloc[::-1].reset_index(drop=True)
            
                model_data = {
                    'ADX': ta.trend.adx(dataset['high'], dataset['low'], dataset['close'], n=14, fillna=True)[-1]
                }
                
                model_data['symbol'] = symbol
                
                return model_data

            print('Getting stocks that need indicators')
            stocks_with_models = get_stocks_with_models()

            indicator_df = pd.DataFrame([get_hist_data_and_add_indicators(stock) for stock in stocks_with_models])
            
            indicator_table_id = 'indicator_table'
            indicator_table_ref = dataset_ref.table(indicator_table_id)
            
            job_config.write_disposition = 'WRITE_TRUNCATE'
            indicator_job = bq_client.load_table_from_dataframe(
                indicator_df,
                indicator_table_ref,
                location='US',
                job_config=job_config
            )

            indicator_job.result()
            print('Indicators ready')

            return 'Success'
            
        else:
            # Market Not Open Today
            pass
    except KeyError as e:
        # Not a weekday
        print(f'Error: {e}')
        pass
