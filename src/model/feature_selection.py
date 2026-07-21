import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

_STRATEGIES = ("f_test", "mutual_info", "rf_embedded", "lasso")

# Numero de folds do TimeSeriesSplit interno do lasso. Fixo (nao exposto no
# construtor) para manter a assinatura publica minima -- ver PLANO_ARQUITETURA.md
# Secao 2, metodo #4: cv deve ser sempre cronologico, nunca KFold aleatorio.
_LASSO_CV_N_SPLITS = 3


def _select_top_k(scores, k, n_features):
    """f_test/mutual_info: mantem as `k` features de maior score. NaN vira
    -inf (score minimo) para nunca ser selecionado por um problema numerico
    do score em si (ex: f_regression numa coluna de variancia zero)."""
    scores = np.nan_to_num(scores, nan=-np.inf)
    k = min(k, n_features)
    return np.sort(np.argsort(scores)[::-1][:k])


def _select_via_embedded_threshold(estimator, X, y):
    """rf_embedded/lasso: SelectFromModel(threshold=None) decide o numero de
    features pelo proprio ajuste. Garante o mesmo invariante das outras 2
    estrategias -- pelo menos 1 feature sempre sobrevive: LassoCV usa um
    threshold fixo de 1e-5 (deteccao de L1 pelo sklearn) que pode zerar TODAS
    as features quando y nao tem nenhuma relacao linear com X (ex: residuo
    ARIMA de uma serie ja bem ajustada linearmente) -- sem esse fallback,
    transform() devolveria shape (n, 0), quebrando MLPRegressor.fit() a
    jusante com 'Found array with 0 feature(s)'. rf_embedded nao sofre desse
    problema por construcao (limiar 'mean' usa comparacao `>=`, entao
    importancias todas zero ainda selecionam todas as features -- nunca
    zero), mas o fallback e aplicado uniformemente por simplicidade e para
    nao depender de uma garantia implicita do sklearn.

    Retorna (selected_indices, fallback_triggered) -- Tarefa 3.6: o segundo
    valor expoe se a selecao final veio do corte normal de SelectFromModel
    ou deste fallback, distincao necessaria para nao confundir "sem sinal
    detectado" com "selecao genuina concentrada em 1 feature"."""
    selector = SelectFromModel(estimator, threshold=None)
    selector.fit(X, y)
    selected = np.where(selector.get_support())[0]
    fallback_triggered = selected.size == 0
    if fallback_triggered:
        fitted = selector.estimator_
        raw_scores = getattr(fitted, "feature_importances_", None)
        if raw_scores is None:
            raw_scores = fitted.coef_
        selected = np.array([np.argmax(np.abs(raw_scores))])
    return np.sort(selected), fallback_triggered


class TimeSeriesFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Seletor de lags para os sistemas hibridos residuais (ver PLANO_ARQUITETURA.md,
    Secao 2). Estrategias suportadas: 'f_test' (filtro linear), 'mutual_info'
    (filtro por teoria da informacao), 'rf_embedded' (embedded, importancia por
    reducao de impureza) e 'lasso' (embedded, regularizacao L1).

    Duas familias de contrato coexistem, ambas convergindo para o mesmo
    atributo publico `selected_indices_` (array ordenado de indices) usado
    por `transform()`:

    - 'f_test'/'mutual_info' (filtros): calculam um score por feature e
      mantem as `k` de maior score -- `k` e um hiperparametro pesquisavel
      pelo GridSearch (`selector__k`).
    - 'rf_embedded'/'lasso' (embedded): delegam a contagem para
      `sklearn.feature_selection.SelectFromModel(threshold=None)` -- o
      numero de features selecionadas emerge do proprio ajuste (corte
      automatico do sklearn: 'mean' das importancias para RF, ~0 para Lasso
      via deteccao de L1 -- ver PLANO_ARQUITETURA.md Secao 1.5). `k` NAO e
      lido por essas duas estrategias (Tarefa 3.1 -- reversao do top-k
      uniforme da Tarefa 3, a pedido do orientador).

    Uso previsto: como o step 'selector' de um sklearn.Pipeline, precedendo o
    estimador (MLPRegressor/SVR) dentro de Additive/SKlearnModel.
    """

    def __init__(self, strategy="f_test", k=5, random_state=None):
        self.strategy = strategy
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.n_features_in_ = X.shape[1]

        if self.strategy == "f_test":
            if self.k <= 0:
                raise ValueError(f"k deve ser um inteiro positivo, recebido: {self.k!r}.")
            scores, _ = f_regression(X, y)
            self.selected_indices_ = _select_top_k(scores, self.k, X.shape[1])
            self.fallback_triggered_ = False
        elif self.strategy == "mutual_info":
            if self.k <= 0:
                raise ValueError(f"k deve ser um inteiro positivo, recebido: {self.k!r}.")
            scores = mutual_info_regression(X, y, random_state=self.random_state)
            self.selected_indices_ = _select_top_k(scores, self.k, X.shape[1])
            self.fallback_triggered_ = False
        elif self.strategy == "rf_embedded":
            estimator = RandomForestRegressor(random_state=self.random_state)
            self.selected_indices_, self.fallback_triggered_ = _select_via_embedded_threshold(estimator, X, y)
        elif self.strategy == "lasso":
            cv = TimeSeriesSplit(n_splits=_LASSO_CV_N_SPLITS)
            estimator = LassoCV(cv=cv, random_state=self.random_state, max_iter=10000)
            self.selected_indices_, self.fallback_triggered_ = _select_via_embedded_threshold(estimator, X, y)
        else:
            raise ValueError(
                f"strategy desconhecida: {self.strategy!r}. "
                f"Use uma de {_STRATEGIES!r}."
            )

        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_indices_]
