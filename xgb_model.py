from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

class XGBModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=random_state)

    def train(self, df):
        df['SMA_200_diff'] = df['close'] - df['SMA_200']
        df['next_return'] = df['close'].pct_change().shift(-1)
        df_model = df.dropna().copy()
        features = ['RSI', 'volatility', 'SMA_200_diff']
        X = df_model[features]
        y = df_model['next_return']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("XGBoost prediction sample:", y_pred[:5])
        return self.model
