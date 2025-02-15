from hmmlearn.hmm import GaussianHMM

class HMMModel:
    def __init__(self, n_components=3, random_state=42):
        self.model = GaussianHMM(n_components=n_components, covariance_type="full",
                                 min_covar=1e-1, n_iter=1000, random_state=random_state)

    def train(self, df):
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df_hmm = df.dropna().copy()
        features = df_hmm[['returns', 'volatility']].values
        self.model.fit(features)
        df_hmm['regime_hmm'] = self.model.predict(features)
        df.loc[df_hmm.index, 'regime_hmm'] = df_hmm['regime_hmm']
        return df
