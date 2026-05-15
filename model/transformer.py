from sklearn.base import BaseEstimator, TransformerMixin


class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X["fornecedor"] + " " + X["descricao"]).values
