import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self):
        self.model = None

    def create_sequences(self, df, timesteps=30):
        data = df[['open', 'high', 'low', 'close', 'volume']].values
        X_seq, y_seq = [], []
        for i in range(len(data) - timesteps):
            X_seq.append(data[i:i+timesteps])
            y_seq.append(data[i+timesteps, 3])  # next close price
        return np.array(X_seq), np.array(y_seq)

    def train(self, df, timesteps=30, epochs=10, batch_size=32):
        if len(df) <= 100:
            print("Not enough data to train LSTM.")
            return None
        X_seq, y_seq = self.create_sequences(df, timesteps)
        self.model = Sequential([
            LSTM(50, input_shape=(X_seq.shape[1], X_seq.shape[2])),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=1)
        print("LSTM model trained for price prediction.")
        return self.model

    def predict(self, df, timesteps=30):
        if self.model is None:
            return None
        data = df[['open', 'high', 'low', 'close', 'volume']].values
        if len(data) < timesteps:
            return None
        sequence = data[-timesteps:]
        sequence = np.expand_dims(sequence, axis=0)
        pred = self.model.predict(sequence)[0][0]
        return pred
