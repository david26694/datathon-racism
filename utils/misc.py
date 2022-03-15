from sklearn.base import BaseEstimator, TransformerMixin


class ToListTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return list(X.flatten())
