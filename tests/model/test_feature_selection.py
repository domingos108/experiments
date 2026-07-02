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
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from model.feature_selection import TimeSeriesFeatureSelector
from tests.model.conftest import FS_DEV_SERIES


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
    def test_fit_transform_succeeds_on_real_series_lags(self, fs_dev_series_train_data, strategy):
        X_train, y_train = fs_dev_series_train_data
        k = min(3, X_train.shape[1])

        selector = TimeSeriesFeatureSelector(strategy=strategy, k=k, random_state=0).fit(X_train, y_train)
        X_transformed = selector.transform(X_train)

        assert selector.selected_indices_.shape == (k,)
        assert np.all((selector.selected_indices_ >= 0) & (selector.selected_indices_ < X_train.shape[1]))
        assert X_transformed.shape == (X_train.shape[0], k)


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
