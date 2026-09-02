"""
`GridSearch(..., estimator_random_state_base=42)` -- seed opt-in, deterministica
POR REPETICAO: a i-esima das `model_exec` repeticoes recebe
`random_state = base + i` no ESTIMADOR (nao no seletor).

Objetivo: tornar as familias MLP (`arima_mlp/*`, `mlp/*`, `model_exec=10`)
REPRODUTIVEIS entre execucoes, SEM colapsar a variancia entre as 10
inicializacoes (CLAUDE.md 3.4 -- a estocasticidade entre reps continua, so
fica pinada). `None` (default) => comportamento byte-a-byte identico ao atual.
"""

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import config
from model import generics, grid_search_exp, single_ml_model_exp
from model.feature_selection import TimeSeriesFeatureSelector


def _gs(tmp_path, monkeypatch, model, model_parameters, base, model_exec=2, **kw):
    monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
    return grid_search_exp.GridSearch(
        single_ml_model_exp.SKlearnModel,
        model,
        model_parameters,
        "fake_seed_exp",
        base,
        "testmodel",
        force=True,
        normalize=True,
        experiment_params={"horizon": 1, "diff_kpss": False},
        model_exec=model_exec,
        **kw,
    )


class TestSeedParamWiring:
    def test_defaults_to_none(self, tmp_path, monkeypatch):
        gs = _gs(tmp_path, monkeypatch, LinearRegression(), {"fit_intercept": [True]}, "airlines.txt")
        assert gs.estimator_random_state_base is None

    def test_none_leaves_estimator_random_state_untouched(self, tmp_path, monkeypatch):
        pipe = Pipeline([("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
                         ("estimator", MLPRegressor())])
        gs = _gs(tmp_path, monkeypatch, pipe, {"selector__k": [3]}, "airlines.txt")
        m = gs._clone_model_for_rep({"selector__k": 3}, rep_index=0)
        assert m.named_steps["estimator"].random_state is None

    def test_pipeline_gets_base_plus_rep_index_on_estimator_step(self, tmp_path, monkeypatch):
        pipe = Pipeline([("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
                         ("estimator", MLPRegressor())])
        gs = _gs(tmp_path, monkeypatch, pipe, {"selector__k": [3]}, "airlines.txt",
                 estimator_random_state_base=42)
        assert gs._clone_model_for_rep({"selector__k": 3}, 0).named_steps["estimator"].random_state == 42
        assert gs._clone_model_for_rep({"selector__k": 3}, 3).named_steps["estimator"].random_state == 45

    def test_selector_random_state_is_NOT_seeded(self, tmp_path, monkeypatch):
        """rf_embedded/rfecv: o RandomForest interno do seletor deve continuar
        random_state=None (PLANO_ARQUITETURA.md 1.5 -- variancia real do nº de
        features e o dado que o orientador quer)."""
        pipe = Pipeline([("selector", TimeSeriesFeatureSelector(strategy="rf_embedded")),
                         ("estimator", MLPRegressor())])
        gs = _gs(tmp_path, monkeypatch, pipe, {}, "airlines.txt", estimator_random_state_base=42)
        m = gs._clone_model_for_rep({}, 0)
        assert m.named_steps["selector"].random_state is None
        assert m.named_steps["estimator"].random_state == 42

    def test_bare_estimator_with_random_state_gets_seeded(self, tmp_path, monkeypatch):
        gs = _gs(tmp_path, monkeypatch, MLPRegressor(), {}, "airlines.txt", estimator_random_state_base=42)
        assert gs._clone_model_for_rep({}, 5).random_state == 47

    def test_bare_svr_with_seed_is_noop_not_error(self, tmp_path, monkeypatch):
        """SVR nao tem random_state -- e deterministico, entao a seed nao se
        aplica: no-op silencioso (o notebook SVR nunca passa o parametro, mas
        se passar, nao deve crashar)."""
        from sklearn.svm import SVR

        gs = _gs(tmp_path, monkeypatch, SVR(), {}, "airlines.txt", estimator_random_state_base=42)
        m = gs._clone_model_for_rep({}, 0)  # nao levanta
        assert "random_state" not in m.get_params()

    def test_seed_on_pipeline_without_estimator_random_state_raises(self, tmp_path, monkeypatch):
        """Achado de code-review: no-op SILENCIOSO e perigoso -- o pesquisador
        acha que esta semeando e nao esta. Se o step nao se chama 'estimator'
        (ou o estimador nao tem random_state), com seed setada, FALHA ALTO."""
        pipe = Pipeline([("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
                         ("mlp", MLPRegressor())])  # step 'mlp', nao 'estimator'
        gs = _gs(tmp_path, monkeypatch, pipe, {"selector__k": [3]}, "airlines.txt",
                 estimator_random_state_base=42)
        with pytest.raises(ValueError, match="random_state"):
            gs._clone_model_for_rep({"selector__k": 3}, 0)

    def test_seed_with_non_sklearn_model_raises_at_construction(self, tmp_path, monkeypatch):
        """Achado de code-review: o ramo is_not_sklearn nunca consulta a seed.
        Passar estimator_random_state_base com um model_class nao-sklearn e um
        erro de uso -- falha na construcao, nao silenciosamente."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        with pytest.raises(ValueError, match="estimator_random_state_base"):
            grid_search_exp.GridSearch(
                object, object(), {"k": [1]}, "e", "airlines.txt", "m",
                experiment_params={"horizon": 1}, estimator_random_state_base=42,
            )


class TestSeedThreadedThroughMultipleBasesWrapper:
    """Achado de code-review: grid_seach_multiple_bases construia GridSearch
    sem repassar a seed -- notebooks que usam o wrapper (nao o laco direto)
    ficariam sem a reprodutibilidade prometida."""

    def test_grid_seach_multiple_bases_forwards_estimator_random_state_base(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        captured = {}
        real_init = grid_search_exp.GridSearch.__init__

        def spy_init(self, *a, **kw):
            captured["base"] = kw.get("estimator_random_state_base", "<ausente>")
            real_init(self, *a, **kw)

        monkeypatch.setattr(grid_search_exp.GridSearch, "__init__", spy_init)
        monkeypatch.setattr(config, "BASE_NAME_LIST", ["airlines.txt"])

        grid_search_exp.grid_seach_multiple_bases(
            single_ml_model_exp.SKlearnModel, LinearRegression(), True,
            {"fit_intercept": [True]}, {"horizon": 1, "diff_kpss": False},
            1, "testmodel", "fake_multi_seed", force=True,
            estimator_random_state_base=42,
        )
        assert captured["base"] == 42


class TestSeedReproducibilityEndToEnd:
    def _run_rmses(self, tmp_path, monkeypatch, base):
        pipe = Pipeline([("selector", TimeSeriesFeatureSelector(strategy="f_test", k=5)),
                         ("estimator", MLPRegressor(activation="logistic", solver="lbfgs", max_iter=200))])
        gs = _gs(tmp_path, monkeypatch, pipe, {"selector__k": [5]}, "airlines.txt",
                 model_exec=4, estimator_random_state_base=base)
        gs.execution()
        saved = generics.open_saved_result(gs.title)
        return [e["experiment"].metrics_results["test_metrics"]["RMSE"] for e in saved]

    def test_two_runs_with_same_seed_are_identical(self, tmp_path, monkeypatch, tmp_path_factory):
        a = self._run_rmses(tmp_path_factory.mktemp("a"), monkeypatch, base=42)
        b = self._run_rmses(tmp_path_factory.mktemp("b"), monkeypatch, base=42)
        assert a == pytest.approx(b)

    def test_variance_between_reps_is_preserved(self, tmp_path, monkeypatch):
        """Seed nao colapsa as 4 reps a um valor unico -- cada rep tem init
        diferente (base+0, base+1, ...), entao os RMSEs variam."""
        rmses = self._run_rmses(tmp_path, monkeypatch, base=42)
        assert len(set(np.round(rmses, 8))) > 1
