import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression, mutual_info_regression

_STRATEGIES = ("f_test", "mutual_info")


class TimeSeriesFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Seletor de lags para os sistemas hibridos residuais (ver PLANO_ARQUITETURA.md,
    Secao 2). Estrategias suportadas nesta fase: 'f_test' (filtro linear) e
    'mutual_info' (filtro por teoria da informacao).

    Uso previsto: como o step 'selector' de um sklearn.Pipeline, precedendo o
    estimador (MLPRegressor/SVR) dentro de Additive/SKlearnModel.
    """

    def __init__(self, strategy="f_test", k=5, random_state=None):
        self.strategy = strategy
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        if self.k <= 0:
            raise ValueError(f"k deve ser um inteiro positivo, recebido: {self.k!r}.")

        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if self.strategy == "f_test":
            scores, _ = f_regression(X, y)
        elif self.strategy == "mutual_info":
            scores = mutual_info_regression(X, y, random_state=self.random_state)
        else:
            raise ValueError(
                f"strategy desconhecida: {self.strategy!r}. "
                f"Use uma de {_STRATEGIES!r}."
            )

        scores = np.nan_to_num(scores, nan=-np.inf)
        k = min(self.k, X.shape[1])
        top_k = np.argsort(scores)[::-1][:k]

        self.selected_indices_ = np.sort(top_k)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_indices_]
