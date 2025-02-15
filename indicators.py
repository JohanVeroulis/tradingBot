import talib

class IndicatorCalculator:
    @staticmethod
    def add_basic_indicators(df):
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], macd_signal, _ = talib.MACD(df['close'])
        df['MACD_signal'] = macd_signal
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'], timeperiod=20)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        return df

    @staticmethod
    def label_regime(df):
        df['SMA_200'] = talib.SMA(df['close'], timeperiod=50)
        df['regime'] = 'neutral'
        df.loc[(df['close'] > df['SMA_200']) & (df['RSI'] > 50), 'regime'] = 'bull'
        df.loc[(df['close'] < df['SMA_200']) & (df['RSI'] < 50), 'regime'] = 'bear'
        return df
