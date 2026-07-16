import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

_STRATEGIES = ("f_test", "mutual_info", "rf_embedded", "lasso")

# Numero de folds do TimeSeriesSplit interno do lasso. Fixo (nao exposto no
# construtor) para manter a assinatura publica minima -- ver PLANO_ARQUITETURA.md
# Secao 2, metodo #4: cv deve ser sempre cronologico, nunca KFold aleatorio.
_LASSO_CV_N_SPLITS = 3


class TimeSeriesFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Seletor de lags para os sistemas hibridos residuais (ver PLANO_ARQUITETURA.md,
    Secao 2). Estrategias suportadas: 'f_test' (filtro linear), 'mutual_info'
    (filtro por teoria da informacao), 'rf_embedded' (embedded, importancia por
    reducao de impureza) e 'lasso' (embedded, regularizacao L1).

    Todas as estrategias compartilham o mesmo contrato: calculam um score por
    feature e mantêm as `k` de maior score -- inclusive as embedded, que aqui
    rankeiam por importancia/coeficiente em vez de usar o threshold automatico
    de SelectFromModel, para que `k` seja um hiperparametro pesquisavel
    uniformemente pelo GridSearch em qualquer estrategia.

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
        elif self.strategy == "rf_embedded":
            estimator = RandomForestRegressor(random_state=self.random_state)
            estimator.fit(X, y)
            scores = estimator.feature_importances_
        elif self.strategy == "lasso":
            cv = TimeSeriesSplit(n_splits=_LASSO_CV_N_SPLITS)
            estimator = LassoCV(cv=cv, random_state=self.random_state, max_iter=10000)
            estimator.fit(X, y)
            scores = np.abs(estimator.coef_)
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
