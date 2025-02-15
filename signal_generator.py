import numpy as np

class SignalGenerator:
    def __init__(self, xgb_model, lstm_model=None):
        self.xgb_model = xgb_model  # Trained XGBRegressor model
        self.lstm_model = lstm_model  # Instance of LSTMModel

    def robust_signal(self, row, prev_row=None):
        bull_rsi_threshold = 55
        bear_rsi_threshold = 65
        bb_buy_proximity = 1.02
        bb_sell_proximity = 0.98

        regime = row.get('regime', 'neutral')
        rsi = row.get('RSI', 50)
        macd = row.get('MACD', 0)
        macd_signal = row.get('MACD_signal', 0)
        close = row.get('close', 0)
        bb_lower = row.get('BB_lower', 0)
        bb_upper = row.get('BB_upper', 0)
        volume = row.get('volume', 0)

        bullish_crossover = False
        bearish_crossover = False
        if prev_row is not None:
            prev_macd = prev_row.get('MACD', 0)
            prev_macd_signal = prev_row.get('MACD_signal', 0)
            bullish_crossover = (prev_macd < prev_macd_signal) and (macd > macd_signal)
            bearish_crossover = (prev_macd > prev_macd_signal) and (macd < macd_signal)

        volume_baseline = 100
        high_volume = volume > (volume_baseline * 1.5)

        signal = "HOLD"
        if regime == 'bull':
            if (rsi < bull_rsi_threshold) or (close < bb_lower * bb_buy_proximity) or bullish_crossover or high_volume:
                signal = 'BUY'
        elif regime == 'bear':
            if (rsi > bear_rsi_threshold) or (close > bb_upper * bb_sell_proximity) or bearish_crossover:
                signal = 'SELL'
        return signal

    def ensemble_signal(self, row, prev_row=None):
        base_signal = self.robust_signal(row, prev_row)
        feature_vector = np.array([row['RSI'], row['volatility'], row['SMA_200_diff']]).reshape(1, -1)
        xgb_pred = self.xgb_model.predict(feature_vector)[0]
        xgb_threshold = 0.001

        lstm_pred = None
        if self.lstm_model is not None:
            lstm_pred = self.lstm_model.predict(row.to_frame().T)  # Adapt as needed

        if base_signal == 'BUY' and xgb_pred > xgb_threshold:
            if lstm_pred is None or lstm_pred > row['close']:
                return 'BUY'
        elif base_signal == 'SELL' and xgb_pred < -xgb_threshold:
            if lstm_pred is None or lstm_pred < row['close']:
                return 'SELL'
        return 'HOLD'
