import time
import pandas as pd

class DataFetcher:
    def __init__(self, session, symbol, interval, days, limit):
        self.session = session
        self.symbol = symbol
        self.interval = interval
        self.days = days
        self.limit = limit

    def fetch_data(self):
        end_time = int(time.time() * 1000)
        start_time = end_time - (self.days * 24 * 60 * 60 * 1000)
        response = self.session.get_kline(
            category='linear',
            symbol=self.symbol,
            interval=self.interval,
            limit=self.limit,
            start=start_time,
            end=end_time
        )
        if response.get('retCode') != 0 or not response.get('result'):
            raise Exception(response.get('retMsg'))
        data = response['result']['list']
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.set_index('timestamp', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df.sort_index(ascending=True, inplace=True)
        # Fill missing values via linear interpolation
        df = df.infer_objects().interpolate()
        return df
