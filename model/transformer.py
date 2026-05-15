from sklearn.base import BaseEstimator, TransformerMixin


class TextCombiner(BaseEstimator, TransformerMixin):
    """Kept for backwards-compat with older model.pkl files."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X["fornecedor"] + " " + X["descricao"]).values


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Extracts a single column from a DataFrame as a 1-D array for TfidfVectorizer."""
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].values
