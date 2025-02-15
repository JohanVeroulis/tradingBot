from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, df):
        df['SMA_200_diff'] = df['close'] - df['SMA_200']
        features = ['RSI', 'volatility', 'SMA_200_diff']
        df = df.dropna().copy()
        df['regime_encoded'] = df['regime'].map({'bear': 0, 'neutral': 1, 'bull': 2})
        X = df[features]
        y = df['regime_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return self.model
