import time
import pandas as pd
import numpy as np
import talib
from pybit.unified_trading import HTTP
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import vectorbt as vbt  # For backtesting and analytics
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import LinAlgError

# ===============================
# Configuration and Initialization
# ===============================
API_KEY = "kt1XXZpqmAOpZKLr7d"  # Your testnet API key
API_SECRET = "dLQwBukEdpYqTbmowOzjlQdwXsJ8iuZM1xkd"  # Your testnet API secret
SYMBOL = "BTCUSDT"
INTERVAL = "15"  # 15-minute candles
DAYS = 1         # For testing – you might want to increase this for robust analysis
LIMIT = 100

# Initialize Bybit session (using testnet mode)
session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=True)

# ===============================
# Data Fetching Function
# ===============================
def fetch_bybit_data(symbol=SYMBOL, interval=INTERVAL, days=DAYS, limit=LIMIT):
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    response = session.get_kline(
        category='linear',
        symbol=symbol,
        interval=interval,
        limit=limit,
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

df = fetch_bybit_data()
print("Data fetched and cleaned:")
print(df.head())

# ===============================
# Technical Indicator Calculations
# ===============================
def add_basic_indicators(df):
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], macd_signal, _ = talib.MACD(df['close'])
    df['MACD_signal'] = macd_signal
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'], timeperiod=20)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    return df

df = add_basic_indicators(df)
df = df.dropna()  # Remove rows with NaNs due to indicator window requirements
print("Data with indicators:")
print(df.head())

# ===============================
# Regime Labeling (Rule-Based)
# ===============================
# For testing, we use a 50-period SMA (even though it's named SMA_200 here)
df['SMA_200'] = talib.SMA(df['close'], timeperiod=50)
df['regime'] = 'neutral'
df.loc[(df['close'] > df['SMA_200']) & (df['RSI'] > 50), 'regime'] = 'bull'
df.loc[(df['close'] < df['SMA_200']) & (df['RSI'] < 50), 'regime'] = 'bear'
print("Data with regime labels (rule-based):")
print(df[['close', 'SMA_200', 'RSI', 'regime']].tail())

# ===============================
# Advanced Regime Classification using HMM
# ===============================
df['returns'] = df['close'].pct_change()
df['volatility'] = df['close'].pct_change().rolling(20).std()
df_hmm = df.dropna().copy()  # Use .copy() to avoid SettingWithCopyWarning
features_hmm = df_hmm[['returns', 'volatility']].values

hmm_model = GaussianHMM(n_components=3, covariance_type="full", min_covar=1e-1, n_iter=1000, random_state=42)
hmm_model.fit(features_hmm)
hidden_states = hmm_model.predict(features_hmm)
df_hmm['regime_hmm'] = hidden_states
df.loc[df_hmm.index, 'regime_hmm'] = df_hmm['regime_hmm']
print("Data with HMM regime labels (regime_hmm):")
print(df[['close', 'returns', 'volatility', 'regime_hmm']].tail())

# ===============================
# Train Random Forest for Regime Detection
# ===============================
df['SMA_200_diff'] = df['close'] - df['SMA_200']
features = ['RSI', 'volatility', 'SMA_200_diff']
target = 'regime'
df_model = df.dropna().copy()  # Use copy to avoid SettingWithCopyWarning
df_model['regime_encoded'] = df_model[target].map({'bear': 0, 'neutral': 1, 'bull': 2})
X = df_model[features]
y = df_model['regime_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest regime detection model trained.")
y_pred = rf_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ===============================
# Train XGBoost for Return Prediction
# ===============================
df_model['next_return'] = df_model['close'].pct_change().shift(-1)
df_model = df_model.dropna()
features_signal = ['RSI', 'volatility', 'SMA_200_diff']
X_signal = df_model[features_signal]
y_signal = df_model['next_return']

X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X_signal, y_signal, test_size=0.2, random_state=42)
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train_sig, y_train_sig)
y_pred_sig = xgb_model.predict(X_test_sig)
print("XGBoost prediction sample:", y_pred_sig[:5])

# ===============================
# Train LSTM for Price Prediction (Optional)
# ===============================
def create_sequences(df, timesteps=30):
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    X_seq, y_seq = [], []
    for i in range(len(data) - timesteps):
        X_seq.append(data[i:i+timesteps])
        y_seq.append(data[i+timesteps, 3])  # Next close price
    return np.array(X_seq), np.array(y_seq)

if len(df) > 100:
    timesteps = 30
    X_seq, y_seq = create_sequences(df, timesteps=timesteps)
    lstm_model = Sequential([
        LSTM(50, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=1)
    print("LSTM model trained for price prediction.")
else:
    lstm_model = None
    print("Not enough data to train LSTM.")

def get_lstm_prediction(df, timesteps=30):
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    if len(data) < timesteps:
        return None
    sequence = data[-timesteps:]
    sequence = np.expand_dims(sequence, axis=0)
    pred = lstm_model.predict(sequence)[0][0]
    return pred

# ===============================
# Ensemble Signal Generation Functions
# ===============================
def robust_signal(row, prev_row=None):
    bull_rsi_threshold = 55   # For bull regime: trigger BUY if RSI is below this threshold
    bear_rsi_threshold = 65   # For bear regime: trigger SELL if RSI is above this threshold
    bb_buy_proximity = 1.02   # BUY if close is within 2% above the lower Bollinger Band
    bb_sell_proximity = 0.98  # SELL if close is within 2% below the upper Bollinger Band

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
    
    volume_baseline = 100  # Adjust this based on your data
    high_volume = volume > (volume_baseline * 1.5)
    
    signal = "HOLD"
    if regime == 'bull':
        if (rsi < bull_rsi_threshold) or (close < bb_lower * bb_buy_proximity) or bullish_crossover or high_volume:
            signal = 'BUY'
    elif regime == 'bear':
        if (rsi > bear_rsi_threshold) or (close > bb_upper * bb_sell_proximity) or bearish_crossover:
            signal = 'SELL'
    return signal

def ensemble_signal(row, prev_row=None):
    base_signal = robust_signal(row, prev_row)
    feature_vector = np.array([row['RSI'], row['volatility'], row['SMA_200_diff']]).reshape(1, -1)
    xgb_pred = xgb_model.predict(feature_vector)[0]
    xgb_threshold = 0.001  # Example threshold for expected return
    
    lstm_pred = get_lstm_prediction(df, timesteps=30) if lstm_model is not None else None
    
    # Ensemble logic: confirm BUY/SELL if both the base signal and ML predictions agree
    if base_signal == 'BUY' and xgb_pred > xgb_threshold:
        if lstm_pred is None or lstm_pred > row['close']:
            return 'BUY'
    elif base_signal == 'SELL' and xgb_pred < -xgb_threshold:
        if lstm_pred is None or lstm_pred < row['close']:
            return 'SELL'
    return 'HOLD'

# ===============================
# Apply Ensemble Signal to Latest Data
# ===============================
latest_row = df.iloc[-1]
prev_row = df.iloc[-2] if len(df) > 1 else None
final_signal = ensemble_signal(latest_row, prev_row)
print("Latest Ensemble Trading Signal for the most recent data:")
print(final_signal)

# ===============================
# Backtesting on Historical Data
# ===============================
# Generate ensemble signals for all rows
ensemble_signals = []
prev = None
for index, row in df.iterrows():
    sig = ensemble_signal(row, prev)
    ensemble_signals.append(sig)
    prev = row
df['ensemble_signal'] = ensemble_signals

# Create boolean series for entries and exits
entries = df['ensemble_signal'] == 'BUY'
exits = df['ensemble_signal'] == 'SELL'

# Backtest using vectorbt with $1,000 initial capital
portfolio = vbt.Portfolio.from_signals(
    close=df['close'],
    entries=entries,
    exits=exits,
    init_cash=1000,
    fees=0.001,
    slippage=0.001
)

# ===============================
# Monitor the Bot Using VectorBT's Built-In Analytics and Plotting
# ===============================
print("Overall Backtest Performance Metrics:")
print(portfolio.stats().to_string())

# # Plot portfolio value over time
# value_fig = portfolio.value().vbt.plot(title="Portfolio Value Over Time")
# value_fig.show()

# # Plot order records (if any)
# orders_fig = portfolio.orders.plot(title="Order Records")
# orders_fig.show()

# Plot trades performance
trades_fig = portfolio.trades.plot(title="Trades Performance")
trades_fig.show()

# Optionally, you could also save these plots or further customize them.
