"""
Testes de src/model/single_ml_model_exp.py -- Tarefa 5 do roadmap
(PLANO_ARQUITETURA.md): confirma, com evidencia real (nao suposicao), que
SKlearnModel e agnostico a identidade de `model` na mesma forma que
hybrid_system_exp.Additive ja e (CLAUDE.md Secao 4.1) -- um
sklearn.Pipeline([('selector', TimeSeriesFeatureSelector(...)),
('estimator', MLPRegressor(...))]) funciona no lugar do MLPRegressor puro,
sem nenhuma mudanca em single_ml_model_exp.py ou grid_search_exp.py.
"""
import config
from model import generics, grid_search_exp, single_ml_model_exp
from model.feature_selection import TimeSeriesFeatureSelector
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


class TestSKlearnModelAcceptsPipeline:
    """fit_predict_ml_schemma (usado por SKlearnModel) so chama
    model.fit(x_train, y_train)/model.predict(x) -- exatamente a mesma
    premissa ja comprovada para Additive. Teste end-to-end com dado real
    (airlines.txt), sem mock, provando que a integracao generaliza para a
    familia MLP single antes de criar os notebooks."""

    def test_pipeline_with_selector_runs_end_to_end_through_sklearn_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        model = Pipeline([
            ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
            ("estimator", MLPRegressor(activation="logistic", solver="lbfgs", max_iter=100)),
        ])

        exec_gs = grid_search_exp.GridSearch(
            single_ml_model_exp.SKlearnModel,
            model,
            {"estimator__hidden_layer_sizes": [5]},
            "fake_experiment",
            "airlines.txt",
            "testmodel",
            force=True,
            normalize=True,
            experiment_params={"diff_kpss": False, "horizon": 1, "type_filter": None},
            model_exec=1,
            use_val_slipt_for_prev=True,
        )
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        fitted_model = saved[0]["experiment"].model
        fitted_selector = fitted_model.named_steps["selector"]

        assert 1 <= fitted_selector.selected_indices_.shape[0] <= fitted_selector.n_features_in_
        assert saved[0]["experiment"].metrics_results["test_metrics"] != {}

    def test_lag_size_auto_resolves_to_same_value_as_arima_mlp_hybrid(self):
        """Tarefa 5, pre-check 2: confirma com dados reais que lag_size='auto'
        resolve para o MESMO valor no contexto MLP single (serie bruta) que
        no hibrido ARIMA-MLP (residuo do ARIMA) -- nao e garantido a priori
        (get_max_lag_to_consider usa PACF sobre a serie que de fato chega em
        open_format_train_val_test, que difere entre os dois contextos), por
        isso medido aqui em vez de assumido. Valores de referencia: medidos
        na Tarefa 5 e documentados em RUNBOOK.md/PLANO_ARQUITETURA.md."""
        from input.input import open_format_train_val_test

        expected_lag_size = {
            "airlines.txt": 20,
            "austres.txt": 1,
            "coloradoRiver.txt": 16,
            "sunspot.txt": 9,
        }
        exec_config_template = {
            "test_size": config.TEST_SIZE,
            "val_size": config.VAL_SIZE,
            "horizon": 1,
            "lag_size": "auto",
            "diff_kpss": False,
            "normalize": True,
            "type_filter": None,
        }

        for series, expected in expected_lag_size.items():
            base_info = open_format_train_val_test(series, dict(exec_config_template))
            assert base_info.lag_size_formated == expected, (
                f"{series}: lag_size='auto' resolveu para {base_info.lag_size_formated}, "
                f"esperado {expected} (mesmo valor ja medido para o hibrido ARIMA-MLP)."
            )


class TestSKlearnModelAcceptsPipelineWithSVR:
    """Tarefa 6: mesma premissa da Tarefa 5 (SKlearnModel e agnostico a
    identidade de `model`), agora com SVR no lugar de MLPRegressor -- a
    unica variavel nova em relacao a Tarefa 5. svr_exec.ipynb (baseline)
    usa diff_kpss=True e model_exec=1 (deterministico, CLAUDE.md Secao 3.4),
    diferente de mlp_exec.ipynb -- testado aqui explicitamente, nao
    reaproveitado por suposicao dos valores do MLP."""

    def test_pipeline_with_selector_runs_end_to_end_through_sklearn_model_svr(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        model = Pipeline([
            ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
            ("estimator", SVR(max_iter=100000)),
        ])

        exec_gs = grid_search_exp.GridSearch(
            single_ml_model_exp.SKlearnModel,
            model,
            {"estimator__C": [10, 100], "estimator__kernel": ["rbf"]},
            "fake_experiment_svr",
            "airlines.txt",
            "testmodelsvr",
            force=True,
            normalize=True,
            experiment_params={"diff_kpss": True, "horizon": 1, "type_filter": None},
            model_exec=1,
            use_val_slipt_for_prev=True,
        )
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        assert len(saved) == 1  # model_exec=1, deterministico -- mesma convencao do baseline SVR
        fitted_model = saved[0]["experiment"].model
        fitted_selector = fitted_model.named_steps["selector"]

        assert 1 <= fitted_selector.selected_indices_.shape[0] <= fitted_selector.n_features_in_
        assert saved[0]["experiment"].metrics_results["test_metrics"] != {}

    def test_lag_size_auto_resolves_to_same_value_with_diff_kpss_true(self):
        """Pre-check 3 da Tarefa 6: lag_size='auto' depende da serie (PACF
        sobre ts_univariate ANTES da diferenciacao KPSS -- ver input.py),
        nao do diff_kpss nem do estimador -- mas medido aqui com
        diff_kpss=True (config real do baseline SVR) em vez de assumido a
        partir do valor ja medido com diff_kpss=False (MLP)."""
        from input.input import open_format_train_val_test

        expected_lag_size = {
            "airlines.txt": 20,
            "austres.txt": 1,
            "coloradoRiver.txt": 16,
            "sunspot.txt": 9,
        }
        exec_config_template = {
            "test_size": config.TEST_SIZE,
            "val_size": config.VAL_SIZE,
            "horizon": 1,
            "lag_size": "auto",
            "diff_kpss": True,
            "normalize": True,
            "type_filter": None,
        }

        for series, expected in expected_lag_size.items():
            base_info = open_format_train_val_test(series, dict(exec_config_template))
            assert base_info.lag_size_formated == expected, (
                f"{series}: lag_size='auto' com diff_kpss=True resolveu para "
                f"{base_info.lag_size_formated}, esperado {expected}."
            )
