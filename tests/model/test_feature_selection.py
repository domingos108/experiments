"""
Testes de TimeSeriesFeatureSelector (estrategias 'f_test' e 'mutual_info').

Cobre o escopo minimo da Tarefa 1 do PLANO_ARQUITETURA.md (Secao 3, item 1):
- corretude de cada estrategia (comparada diretamente contra o sklearn puro)
- compatibilidade com o contrato sklearn (clone/get_params/set_params), que e
  exatamente o que GridSearch._search_params() usa em producao
- Zero Data Leakage: fit() nunca pode ser influenciado por linhas fora do
  fold de treino que lhe foi passado
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, TimeSeriesSplit

from model.feature_selection import TimeSeriesFeatureSelector
from tests.model.conftest import FS_DEV_SERIES, load_fs_dev_series_train_data


def _synthetic_linear_data(n_samples=60, n_features=8, informative_idx=(0, 3), seed=0):
    """Alvo e combinacao linear de duas colunas + ruido; demais colunas sao puro ruido."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = 5.0 * X[:, informative_idx[0]] - 3.0 * X[:, informative_idx[1]] + 0.01 * rng.normal(size=n_samples)
    return X, y


def _synthetic_nonlinear_data(n_samples=200, n_features=8, informative_idx=0, seed=0):
    """Alvo e uma funcao nao-monotonica (quadratica) de uma unica coluna informativa;
    demais colunas sao ruido puro, sem qualquer relacao com o alvo."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    y = X[:, informative_idx] ** 2 + 0.05 * rng.normal(size=n_samples)
    return X, y


def _synthetic_sparse_linear_data(n_samples=200, n_features=10, n_informative=2, seed=0):
    """Alvo e combinacao linear de `n_informative` colunas (as primeiras);
    controla o grau de esparsidade do sinal para testar contagem variavel de
    features selecionadas por SelectFromModel (Tarefa 3.1)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    coef = np.zeros(n_features)
    coef[:n_informative] = rng.uniform(2, 5, size=n_informative)
    y = X @ coef + 0.1 * rng.normal(size=n_samples)
    return X, y


class TestStrategyValidation:
    def test_unknown_strategy_raises_on_fit(self):
        X, y = _synthetic_linear_data()
        selector = TimeSeriesFeatureSelector(strategy="does_not_exist", k=3)
        with pytest.raises(ValueError, match="does_not_exist"):
            selector.fit(X, y)

    @pytest.mark.parametrize("invalid_k", [0, -1, -5])
    def test_non_positive_k_raises_on_fit(self, invalid_k):
        X, y = _synthetic_linear_data()
        selector = TimeSeriesFeatureSelector(strategy="f_test", k=invalid_k)
        with pytest.raises(ValueError, match="k"):
            selector.fit(X, y)

    @pytest.mark.parametrize("strategy", ["rf_embedded", "lasso", "rfecv"])
    @pytest.mark.parametrize("unused_k", [0, -1, -5])
    def test_non_positive_k_does_not_raise_for_embedded_strategies(self, strategy, unused_k):
        """Achado de code-review na Tarefa 3.1: k NAO e lido por rf_embedded/
        lasso/rfecv (docstring da classe) -- k=0/negativo nao pode mais
        disparar ValueError para essas estrategias, senao a mensagem do
        docstring ('k NAO e lido') fica inconsistente com o comportamento
        real."""
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))
        selector = TimeSeriesFeatureSelector(strategy=strategy, k=unused_k, random_state=0)

        selector.fit(X, y)  # nao deve levantar

        assert selector.selected_indices_.shape[0] >= 1


class TestFTestStrategy:
    def test_selects_same_indices_as_plain_sklearn_selectkbest(self):
        X, y = _synthetic_linear_data(informative_idx=(0, 3))
        k = 2

        expected = SelectKBest(score_func=f_regression, k=k).fit(X, y)
        expected_indices = np.sort(np.where(expected.get_support())[0])

        selector = TimeSeriesFeatureSelector(strategy="f_test", k=k).fit(X, y)

        assert np.array_equal(selector.selected_indices_, expected_indices)

    def test_transform_keeps_only_selected_columns(self):
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))
        selector = TimeSeriesFeatureSelector(strategy="f_test", k=2).fit(X, y)

        X_transformed = selector.transform(X)

        assert X_transformed.shape == (X.shape[0], 2)
        assert np.array_equal(X_transformed, X[:, selector.selected_indices_])


class TestMutualInfoStrategy:
    def test_ranks_nonlinear_informative_feature_above_pure_noise(self):
        # Feature 0 tem relacao quadratica (nao-linear) com y; as demais sao ruido puro.
        # f_test (linear) tende a nao rankear feature 0 no topo; mutual_info deve.
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        selector = TimeSeriesFeatureSelector(strategy="mutual_info", k=1, random_state=0).fit(X, y)

        assert selector.selected_indices_[0] == 0

    def test_matches_plain_sklearn_mutual_info_regression_ranking(self):
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)
        k = 3

        expected_scores = mutual_info_regression(X, y, random_state=0)
        expected_indices = np.sort(np.argsort(expected_scores)[::-1][:k])

        selector = TimeSeriesFeatureSelector(strategy="mutual_info", k=k, random_state=0).fit(X, y)

        assert np.array_equal(selector.selected_indices_, expected_indices)


class TestRfEmbeddedStrategy:
    """Tarefa 3.1: rf_embedded usa SelectFromModel(threshold=None) -- o numero
    de features selecionadas emerge do ajuste (corte no 'mean' das
    importancias, resolvido automaticamente pelo sklearn para estimadores
    sem atributo `penalty` -- ver PLANO_ARQUITETURA.md Secao 1.5), nao mais
    um `k` fixo de grid search."""

    def test_ranks_nonlinear_informative_feature_above_pure_noise(self):
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        selector = TimeSeriesFeatureSelector(strategy="rf_embedded", random_state=0).fit(X, y)

        assert 0 in selector.selected_indices_.tolist()

    def test_deterministic_with_fixed_random_state(self):
        """model_exec=10 repete o fit varias vezes na mesma configuracao --
        precisa ser reprodutivel para a comparacao de hiperparametros do
        GridSearch fazer sentido."""
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        selector_a = TimeSeriesFeatureSelector(strategy="rf_embedded", random_state=42).fit(X, y)
        selector_b = TimeSeriesFeatureSelector(strategy="rf_embedded", random_state=42).fit(X, y)

        assert np.array_equal(selector_a.selected_indices_, selector_b.selected_indices_)

    def test_matches_plain_sklearn_select_from_model_with_default_threshold(self):
        """Corretude: o resultado deve ser identico a usar SelectFromModel
        puro do sklearn com threshold=None (nao uma reimplementacao propria
        do corte por importancia)."""
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        expected = SelectFromModel(
            RandomForestRegressor(random_state=0), threshold=None
        ).fit(X, y)
        expected_indices = np.sort(np.where(expected.get_support())[0])

        selector = TimeSeriesFeatureSelector(strategy="rf_embedded", random_state=0).fit(X, y)

        assert np.array_equal(selector.selected_indices_, expected_indices)

    def test_k_parameter_is_ignored_for_this_strategy(self):
        """`k` deixa de ser lido por rf_embedded/lasso (Tarefa 3.1) -- valores
        diferentes de `k` nao podem alterar o resultado, que agora e
        inteiramente decidido pelo threshold do SelectFromModel."""
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        selector_k1 = TimeSeriesFeatureSelector(strategy="rf_embedded", k=1, random_state=0).fit(X, y)
        selector_k7 = TimeSeriesFeatureSelector(strategy="rf_embedded", k=7, random_state=0).fit(X, y)

        assert np.array_equal(selector_k1.selected_indices_, selector_k7.selected_indices_)

    def test_number_of_selected_features_varies_with_data_sparsity(self):
        """Requisito central da Tarefa 3.1: a contagem de features
        selecionadas deve variar conforme os dados, nao ser fixa. Dados com
        sinal concentrado em poucas colunas devem selecionar menos features
        que dados com sinal espalhado por muitas colunas."""
        X_sparse, y_sparse = _synthetic_sparse_linear_data(n_features=10, n_informative=2, seed=0)
        X_dense, y_dense = _synthetic_sparse_linear_data(n_features=10, n_informative=8, seed=0)

        selector_sparse = TimeSeriesFeatureSelector(strategy="rf_embedded", random_state=0).fit(X_sparse, y_sparse)
        selector_dense = TimeSeriesFeatureSelector(strategy="rf_embedded", random_state=0).fit(X_dense, y_dense)

        assert selector_sparse.selected_indices_.shape[0] != selector_dense.selected_indices_.shape[0]

    def test_fallback_never_triggers_for_rf_embedded_even_with_zero_signal(self):
        """Tarefa 3.6: docstring de _select_via_embedded_threshold ja
        documentava que rf_embedded nao sofre do problema de zero-features
        (threshold 'mean' usa `>=`) -- fallback_triggered_ deve confirmar
        isso mecanicamente mesmo no mesmo cenario degenerado (y constante)
        que forca o Lasso a zerar tudo."""
        X = np.random.RandomState(0).normal(size=(100, 10))
        y = np.zeros(100)

        selector = TimeSeriesFeatureSelector(strategy="rf_embedded", random_state=0).fit(X, y)

        assert selector.fallback_triggered_ is False


class TestLassoStrategy:
    def test_selects_the_two_truly_linear_informative_features(self):
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))

        selector = TimeSeriesFeatureSelector(strategy="lasso", random_state=0).fit(X, y)

        assert set(selector.selected_indices_.tolist()) == {0, 3}

    def test_matches_plain_sklearn_select_from_model_with_default_threshold(self):
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))
        cv = TimeSeriesSplit(n_splits=3)

        expected = SelectFromModel(
            LassoCV(cv=cv, random_state=0, max_iter=10000), threshold=None
        ).fit(X, y)
        expected_indices = np.sort(np.where(expected.get_support())[0])

        selector = TimeSeriesFeatureSelector(strategy="lasso", random_state=0).fit(X, y)

        assert np.array_equal(selector.selected_indices_, expected_indices)

    def test_k_parameter_is_ignored_for_this_strategy(self):
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))

        selector_k1 = TimeSeriesFeatureSelector(strategy="lasso", k=1, random_state=0).fit(X, y)
        selector_k7 = TimeSeriesFeatureSelector(strategy="lasso", k=7, random_state=0).fit(X, y)

        assert np.array_equal(selector_k1.selected_indices_, selector_k7.selected_indices_)

    def test_number_of_selected_features_varies_with_data_sparsity(self):
        X_sparse, y_sparse = _synthetic_sparse_linear_data(n_features=10, n_informative=2, seed=2)
        X_dense, y_dense = _synthetic_sparse_linear_data(n_features=10, n_informative=8, seed=2)

        selector_sparse = TimeSeriesFeatureSelector(strategy="lasso", random_state=0).fit(X_sparse, y_sparse)
        selector_dense = TimeSeriesFeatureSelector(strategy="lasso", random_state=0).fit(X_dense, y_dense)

        assert selector_sparse.selected_indices_.shape[0] != selector_dense.selected_indices_.shape[0]

    def test_fallback_triggered_is_false_on_a_normal_selectfrommodel_cut(self):
        """Tarefa 3.6: quando SelectFromModel corta normalmente (>=1 feature
        sobrevive ao threshold por si so), fallback_triggered_ deve ser False
        -- a selecao veio do Lasso, nao do fallback de zero-features."""
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))

        selector = TimeSeriesFeatureSelector(strategy="lasso", random_state=0).fit(X, y)

        assert selector.fallback_triggered_ is False

    def test_falls_back_to_one_feature_when_lasso_zeroes_every_coefficient(self):
        """Bug real encontrado por code-review (angulo cross-file) na Tarefa
        3.1: SelectFromModel(LassoCV(...), threshold=None) usa um limiar fixo
        de 1e-5 para estimadores L1 -- se o alvo nao tem NENHUMA relacao
        linear com X (ex: residuo ARIMA de uma serie ja bem ajustada
        linearmente, ou -- caso extremo reproduzido aqui -- y constante),
        TODOS os coeficientes do Lasso podem zerar, e SelectFromModel
        seleciona 0 features. `transform()` retornaria shape (n, 0), que
        quebra MLPRegressor.fit() com
        'Found array with 0 feature(s)... while a minimum of 1 is required'.
        rf_embedded NAO tem esse problema (limiar 'mean' usa `>=`, entao o
        caso degenerado de importancias todas zero ainda satisfaz o proprio
        threshold e seleciona todas as features, nunca zero) -- confirmado
        por reproducao direta contra o sklearn instalado antes deste teste.
        A correcao deve garantir o mesmo invariante de todas as outras 3
        estrategias: pelo menos 1 feature sempre sobrevive."""
        X = np.random.RandomState(0).normal(size=(100, 10))
        y = np.zeros(100)  # sem NENHUMA relacao com X -- forca Lasso a zerar tudo

        selector = TimeSeriesFeatureSelector(strategy="lasso", random_state=0).fit(X, y)

        assert selector.selected_indices_.shape[0] >= 1
        X_transformed = selector.transform(X)
        assert X_transformed.shape[1] >= 1
        # Tarefa 3.6: expoe que a selecao acima veio do fallback, nao de um
        # corte genuino do SelectFromModel -- distincao cientificamente
        # relevante (ausencia de sinal linear mascarada de selecao real).
        assert selector.fallback_triggered_ is True

    def test_cv_is_time_series_split_never_kfold(self, monkeypatch):
        """Regra nao-negociavel do PLANO_ARQUITETURA.md (Secao 2, metodo #4):
        o cv interno do LassoCV NUNCA pode ser um KFold aleatorio -- espiona
        o construtor de LassoCV e confirma o tipo exato do cv recebido."""
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))
        captured = {}

        real_lasso_cv = LassoCV

        def spy_lasso_cv(*args, **kwargs):
            captured["cv"] = kwargs.get("cv")
            return real_lasso_cv(*args, **kwargs)

        monkeypatch.setattr("model.feature_selection.LassoCV", spy_lasso_cv)

        TimeSeriesFeatureSelector(strategy="lasso", k=2, random_state=0).fit(X, y)

        assert isinstance(captured["cv"], TimeSeriesSplit)
        assert not isinstance(captured["cv"], KFold)

    def test_fit_only_ever_scores_the_exact_rows_it_was_given(self, monkeypatch):
        """Mesma prova mecanica de Zero Data Leakage usada para f_test,
        aplicada ao lasso -- que e o outro metodo com CV interna, portanto o
        de maior risco de vazamento indireto."""
        X_train, y_train = _synthetic_linear_data(n_samples=50, n_features=6, seed=1)
        captured = {}

        real_lasso_cv = LassoCV

        class SpyLassoCV(real_lasso_cv):
            def fit(self, X, y, **kwargs):
                captured["X"] = np.array(X, copy=True)
                captured["y"] = np.array(y, copy=True)
                return super().fit(X, y, **kwargs)

        monkeypatch.setattr("model.feature_selection.LassoCV", SpyLassoCV)

        TimeSeriesFeatureSelector(strategy="lasso", k=2, random_state=0).fit(X_train, y_train)

        assert captured["X"].shape == X_train.shape
        assert np.array_equal(captured["X"], X_train)
        assert np.array_equal(captured["y"], y_train)


class TestRfecvStrategy:
    """Tarefa 4: rfecv usa RFECV(RandomForestRegressor, cv=TimeSeriesSplit(...),
    min_features_to_select=1) -- eliminacao recursiva guiada por CV cronologica.
    Diferente de rf_embedded/lasso, min_features_to_select=1 e um piso GARANTIDO
    pelo proprio RFECV (nunca retorna 0 features) -- nao precisa do mecanismo de
    fallback da Tarefa 3.1. Mas RFECV exige >=2 features de entrada (RFE nao faz
    sentido com 1 unica candidata) -- TimeSeriesFeatureSelector precisa de uma
    guarda propria para esse caso (decisao confirmada com o pesquisador antes de
    implementar, Tarefa 4)."""

    def test_ranks_nonlinear_informative_feature_above_pure_noise(self):
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        selector = TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X, y)

        assert 0 in selector.selected_indices_.tolist()

    def test_deterministic_with_fixed_random_state(self):
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        selector_a = TimeSeriesFeatureSelector(strategy="rfecv", random_state=42).fit(X, y)
        selector_b = TimeSeriesFeatureSelector(strategy="rfecv", random_state=42).fit(X, y)

        assert np.array_equal(selector_a.selected_indices_, selector_b.selected_indices_)

    def test_matches_plain_sklearn_rfecv(self):
        """Corretude: o resultado deve ser identico a usar RFECV puro do
        sklearn com os mesmos hiperparametros (nao uma reimplementacao propria)."""
        from sklearn.feature_selection import RFECV

        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        expected = RFECV(
            RandomForestRegressor(random_state=0),
            cv=TimeSeriesSplit(n_splits=3),
            step=1,
            min_features_to_select=1,
        ).fit(X, y)
        expected_indices = np.sort(np.where(expected.support_)[0])

        selector = TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X, y)

        assert np.array_equal(selector.selected_indices_, expected_indices)

    def test_k_parameter_is_ignored_for_this_strategy(self):
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        selector_k1 = TimeSeriesFeatureSelector(strategy="rfecv", k=1, random_state=0).fit(X, y)
        selector_k7 = TimeSeriesFeatureSelector(strategy="rfecv", k=7, random_state=0).fit(X, y)

        assert np.array_equal(selector_k1.selected_indices_, selector_k7.selected_indices_)

    def test_number_of_selected_features_varies_with_data_sparsity(self):
        X_sparse, y_sparse = _synthetic_sparse_linear_data(n_features=10, n_informative=2, seed=0)
        X_dense, y_dense = _synthetic_sparse_linear_data(n_features=10, n_informative=8, seed=0)

        selector_sparse = TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X_sparse, y_sparse)
        selector_dense = TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X_dense, y_dense)

        assert selector_sparse.selected_indices_.shape[0] != selector_dense.selected_indices_.shape[0]

    def test_falls_back_to_the_single_feature_when_only_one_candidate_exists(self):
        """Achado do brainstorming da Tarefa 4 (reproduzido com sklearn real
        antes de implementar): RFECV levanta ValueError com 'Found array with
        1 feature(s)... while a minimum of 2 is required' quando X tem 1 unica
        coluna -- caso real de austres.txt (N_Features_Total=1, Tarefa 3.2).
        A guarda decidida com o pesquisador: manter a unica feature
        trivialmente, sem chamar RFECV, mesmo invariante das outras 4
        estrategias (pelo menos 1 feature sempre sobrevive, nunca crasha)."""
        X = np.random.RandomState(0).normal(size=(80, 1))
        y = np.random.RandomState(1).normal(size=80)

        selector = TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X, y)

        assert selector.selected_indices_.tolist() == [0]
        assert selector.fallback_triggered_ is True

    def test_falls_back_using_real_austres_data_with_lag_size_one(self):
        """Fecha a lacuna apontada por code-review (angulo line-by-line,
        Tarefa 4): o cenario real que motivou a guarda (austres.txt,
        N_Features_Total=1 via lag_size='auto', Tarefa 3.2) precisa ser
        exercitado end-to-end pelo pipeline de producao
        (open_format_train_val_test -> create_windowing), nao so por dado
        sintetico -- o fixture generico (fs_dev_series_train_data) usa
        k_lags=5 fixo e nunca aciona esta guarda."""
        X_train, y_train = load_fs_dev_series_train_data("austres", k_lags=1)
        assert X_train.shape[1] == 1  # confirma que o cenario real foi reproduzido

        selector = TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X_train, y_train)

        assert selector.selected_indices_.tolist() == [0]
        assert selector.fallback_triggered_ is True

    def test_raises_clear_error_instead_of_silent_invalid_index_when_zero_features(self):
        """Achado de code-review (angulos line-by-line + altitude, Tarefa 4):
        a guarda original testava `X.shape[1] < 2`, cobrindo tanto 0 quanto 1
        feature com o mesmo `np.array([0])` -- para 0 features esse indice
        nao existe, e o erro so apareceria depois, dentro de transform()
        (IndexError), longe da causa raiz. 0 features e um estado
        estruturalmente invalido (nao acontece via create_windowing hoje,
        lag_size sempre >=1) -- deve falhar alto e claro, nao ser tratado
        como o mesmo caso de negocio legitimo que 1 feature."""
        X = np.zeros((80, 0))
        y = np.zeros(80)

        with pytest.raises(ValueError, match="0 feature"):
            TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X, y)

    def test_fallback_triggered_is_false_when_rfecv_genuinely_runs(self):
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)

        selector = TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X, y)

        assert selector.fallback_triggered_ is False

    def test_cv_is_time_series_split_never_kfold(self, monkeypatch):
        """Regra nao-negociavel do PLANO_ARQUITETURA.md (Secao 2, metodo #5)
        e reforcada explicitamente no prompt da Tarefa 4: RFECV e o metodo
        mais caro e mais facil de configurar errado -- espiona RFECV e
        confirma o tipo exato do cv recebido."""
        X, y = _synthetic_nonlinear_data(n_features=8, informative_idx=0)
        captured = {}

        from sklearn.feature_selection import RFECV as real_RFECV

        def spy_rfecv(*args, **kwargs):
            captured["cv"] = kwargs.get("cv")
            return real_RFECV(*args, **kwargs)

        monkeypatch.setattr("model.feature_selection.RFECV", spy_rfecv)

        TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X, y)

        assert isinstance(captured["cv"], TimeSeriesSplit)
        assert not isinstance(captured["cv"], KFold)

    def test_fit_only_ever_scores_the_exact_rows_it_was_given(self, monkeypatch):
        """Mesma prova mecanica de Zero Data Leakage usada para f_test/lasso,
        aplicada ao rfecv -- o metodo mais caro e com CV interna (RFECV chama
        RandomForestRegressor.fit() muitas vezes: por fold x por rodada de
        eliminacao), portanto o de maior risco de vazamento indireto. Captura
        TODAS as chamadas internas (nao so a ultima) e prova que nenhuma
        jamais viu uma linha fora de X_train/y_train."""
        X_train, y_train = _synthetic_linear_data(n_samples=50, n_features=6, seed=1)
        calls = []

        real_random_forest = RandomForestRegressor

        class SpyRandomForestRegressor(real_random_forest):
            def fit(self, X, y, **kwargs):
                calls.append((np.array(X, copy=True), np.array(y, copy=True)))
                return super().fit(X, y, **kwargs)

        monkeypatch.setattr("model.feature_selection.RandomForestRegressor", SpyRandomForestRegressor)

        TimeSeriesFeatureSelector(strategy="rfecv", random_state=0).fit(X_train, y_train)

        assert len(calls) > 0
        y_train_rounded = set(np.round(y_train, 8).tolist())
        for captured_X, captured_y in calls:
            # cada fit interno recebe um SUBCONJUNTO de linhas de X_train/y_train
            # (o fold da TimeSeriesSplit) -- nunca mais linhas, nunca outras
            # linhas. Colunas podem ser MENOS que X_train (RFE elimina
            # features ao longo das rodadas -- isso e esperado, nao vazamento).
            assert captured_X.shape[0] <= X_train.shape[0]
            assert captured_X.shape[1] <= X_train.shape[1]
            assert set(np.round(captured_y, 8).tolist()) <= y_train_rounded


class TestSklearnContract:
    def test_get_params_returns_exactly_constructor_args(self):
        selector = TimeSeriesFeatureSelector(strategy="f_test", k=4)

        params = selector.get_params()

        assert params == {"strategy": "f_test", "k": 4, "random_state": None}

    def test_clone_and_set_params_mirrors_gridsearch_usage(self):
        # Replica exatamente o padrao usado em GridSearch._search_params():
        #   clone(self.model).set_params(**params)
        base_selector = TimeSeriesFeatureSelector(strategy="f_test", k=4)

        cloned = clone(base_selector).set_params(**{"strategy": "mutual_info", "k": 2})

        assert cloned.strategy == "mutual_info"
        assert cloned.k == 2
        # o clone nao deve ter herdado nenhum estado de fit do original
        assert not hasattr(cloned, "selected_indices_")


class TestRealSeriesData:
    """Fecha a lacuna apontada no pre-check da Tarefa 2: os testes de
    f_test/mutual_info precisam cobrir as 4 series de FS_DEV_SERIES com dados
    reais (via input.open_format_train_val_test), nao apenas dados sinteticos."""

    def test_fs_dev_series_is_exactly_the_four_series_from_plano_arquitetura(self):
        assert FS_DEV_SERIES == ["airlines", "austres", "coloradoRiver", "sunspot"]

    @pytest.mark.parametrize("strategy", ["f_test", "mutual_info"])
    def test_fit_transform_succeeds_on_real_series_lags_fixed_k(self, fs_dev_series_train_data, strategy):
        X_train, y_train = fs_dev_series_train_data
        k = min(3, X_train.shape[1])

        selector = TimeSeriesFeatureSelector(strategy=strategy, k=k, random_state=0).fit(X_train, y_train)
        X_transformed = selector.transform(X_train)

        assert selector.selected_indices_.shape == (k,)
        assert np.all((selector.selected_indices_ >= 0) & (selector.selected_indices_ < X_train.shape[1]))
        assert X_transformed.shape == (X_train.shape[0], k)

    @pytest.mark.parametrize("strategy", ["rf_embedded", "lasso", "rfecv"])
    def test_fit_transform_succeeds_on_real_series_lags_variable_count(self, fs_dev_series_train_data, strategy):
        """rf_embedded/lasso (Tarefa 3.1) e rfecv (Tarefa 4): a contagem
        selecionada e decidida pelo proprio ajuste (SelectFromModel/RFECV),
        nao por `k` -- so validamos que fica dentro do intervalo valido
        [1, n_features_total] e que transform() e consistente com
        selected_indices_. Roda inclusive contra austres.txt (via
        fs_dev_series_train_data, k_lags=5 fixo no fixture -- nao aciona a
        guarda de 1-feature deste teste, que e coberta separadamente em
        TestRfecvStrategy)."""
        X_train, y_train = fs_dev_series_train_data

        selector = TimeSeriesFeatureSelector(strategy=strategy, random_state=0).fit(X_train, y_train)
        X_transformed = selector.transform(X_train)
        n_selected = selector.selected_indices_.shape[0]

        assert 1 <= n_selected <= X_train.shape[1]
        assert np.all((selector.selected_indices_ >= 0) & (selector.selected_indices_ < X_train.shape[1]))
        assert X_transformed.shape == (X_train.shape[0], n_selected)


class TestZeroDataLeakage:
    def test_fit_only_ever_scores_the_exact_rows_it_was_given(self, monkeypatch):
        """
        Prova mecanica (nao estatistica) de Zero Data Leakage: substitui a
        funcao de score interna por um espiao que captura os arrays
        exatamente como chegam. Se fit() algum dia passar a montar/concatenar
        dados extras (val/test) antes de pontuar, este teste falha
        imediatamente -- independente de qualquer coincidencia estatistica.
        """
        X_train, y_train = _synthetic_linear_data(n_samples=50, n_features=6, seed=1)
        captured = {}

        def fake_f_regression(X, y):
            captured["X"] = np.array(X, copy=True)
            captured["y"] = np.array(y, copy=True)
            return f_regression(X, y)

        monkeypatch.setattr("model.feature_selection.f_regression", fake_f_regression)

        TimeSeriesFeatureSelector(strategy="f_test", k=2).fit(X_train, y_train)

        assert captured["X"].shape == X_train.shape
        assert np.array_equal(captured["X"], X_train)
        assert np.array_equal(captured["y"], y_train)

    def test_fit_does_not_retain_reference_to_input_arrays(self):
        """fit() nao deve guardar X/y crus -- so os artefatos derivados (scores/indices).
        Isso impede que um transform() posterior "vaze" dados de fit indevidamente."""
        X, y = _synthetic_linear_data()
        selector = TimeSeriesFeatureSelector(strategy="f_test", k=2).fit(X, y)

        forbidden_attrs = {"X", "y", "X_", "y_", "X_train", "y_train"}
        assert forbidden_attrs.isdisjoint(vars(selector).keys())


class TestFallbackTriggeredAttribute:
    """Tarefa 3.6: expoe se a selecao final veio do fallback de zero-features
    (Tarefa 3.1) ou de um corte genuino -- necessario para distinguir
    "Lasso concentrou o sinal em 1 feature" de "Lasso nao achou sinal
    nenhum e o fallback deterministico mascarou isso como selecao"."""

    @pytest.mark.parametrize("strategy", ["f_test", "mutual_info"])
    def test_always_false_for_filter_strategies(self, strategy):
        """f_test/mutual_info nao tem conceito de fallback -- sempre mantem
        exatamente k features por construcao, entao o atributo existe (para
        uso uniforme por quem consome as 5 estrategias) mas e sempre False.
        rfecv tem sua PROPRIA semantica de fallback (guarda de <2 features,
        nao "corte zerado" como rf_embedded/lasso) -- coberta separadamente
        em TestRfecvStrategy, nao duplicada aqui."""
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))

        selector = TimeSeriesFeatureSelector(strategy=strategy, k=2, random_state=0).fit(X, y)

        assert selector.fallback_triggered_ is False


class TestNFeaturesInAttribute:
    """Tarefa 3.1, Parte C: o script de extracao de features selecionadas
    (src/utils/export_selected_features.py) precisa saber o total de
    features candidatas (n_features_total), nao so as selecionadas -- exposto
    de forma uniforme nas 5 estrategias via `n_features_in_` (convencao
    sklearn), setado a partir de X.shape[1] em fit()."""

    @pytest.mark.parametrize("strategy", ["f_test", "mutual_info", "rf_embedded", "lasso", "rfecv"])
    def test_n_features_in_matches_input_column_count(self, strategy):
        X, y = _synthetic_linear_data(n_features=8, informative_idx=(0, 3))

        selector = TimeSeriesFeatureSelector(strategy=strategy, k=2, random_state=0).fit(X, y)

        assert selector.n_features_in_ == 8
