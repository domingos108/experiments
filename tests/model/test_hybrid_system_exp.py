"""
Testes de src/model/hybrid_system_exp.py -- Tarefa 7 do roadmap
(PLANO_ARQUITETURA.md): confirma, com evidencia real (nao suposicao), que a
combinacao Additive + Pipeline([selector, SVR]) funciona sem nenhuma mudanca
de codigo -- fecha a matriz completa de 5 familias (ARIMA, ARIMA-MLP, MLP,
SVR, ARIMA-SVR) x 5 metodos de FS. Additive ja era comprovadamente agnostico
a identidade de `model` com MLPRegressor (Tarefas 2/3); esta tarefa e a
primeira vez que SVR e testado dentro de Additive.
"""
import shutil
from pathlib import Path

import config
from model import generics, grid_search_exp, hybrid_system_exp
from model.feature_selection import TimeSeriesFeatureSelector
from model.hybrid_system_exp import input_linear_info
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


class TestAdditiveAcceptsPipelineWithSVR:
    """fit_predict de Additive so chama generics.fit_predict_model(self.model, ...),
    que por sua vez so faz model.fit/model.predict -- exatamente a mesma
    premissa ja comprovada com MLPRegressor (Tarefas 2/3) e com SVR dentro de
    SKlearnModel (Tarefa 6). Teste end-to-end com dado real (airlines.txt),
    sem mock, copiando o .pkl real do ARIMA (Additive exige o modelo linear
    pre-treinado sob o mesmo experiment_id)."""

    def test_pipeline_with_selector_runs_end_to_end_through_additive_svr(self, tmp_path, monkeypatch):
        real_model_data_path = config.MODEL_DATA_PATH
        tmp_chamados = tmp_path / "chamados"
        tmp_chamados.mkdir()
        shutil.copy(Path(real_model_data_path) / "chamados" / "airlines_1arima.pkl", tmp_chamados)

        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        from utils.copy_pretrained_linear_model import copy_pretrained_linear_model

        copy_pretrained_linear_model(
            source_experiment_id="chamados",
            dest_experiment_id="fake_experiment_arimasvr",
            series_list=["airlines.txt"],
            linear_model_name="1arima",
        )

        model = Pipeline([
            ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
            ("estimator", SVR(max_iter=100000)),
        ])

        exec_gs = grid_search_exp.GridSearch(
            hybrid_system_exp.Additive,
            model,
            {"estimator__C": [10, 100], "estimator__kernel": ["rbf"]},
            "fake_experiment_arimasvr",
            "airlines.txt",
            "testmodelarimasvr",
            force=True,
            normalize=True,
            experiment_params={"linear_model_name": "1arima", "diff_kpss": False, "horizon": 1},
            model_exec=1,
            use_val_slipt_for_prev=True,
        )
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        assert len(saved) == 1  # model_exec=1, deterministico -- mesma convencao do baseline
        fitted_model = saved[0]["experiment"].model
        fitted_selector = fitted_model.named_steps["selector"]

        assert 1 <= fitted_selector.selected_indices_.shape[0] <= fitted_selector.n_features_in_
        assert saved[0]["experiment"].metrics_results["test_metrics"] != {}

    def test_lag_size_auto_resolves_to_same_value_as_arima_mlp_hybrid(self):
        """Pre-check 2 da Tarefa 7: lag_size='auto' no contexto hibrido
        ARIMA-SVR (residuo do ARIMA, diff_kpss=False -- mesma config de
        arima_svr.ipynb) medido diretamente, nao assumido igual a familia
        SVR single (que usa a serie bruta com diff_kpss=True) nem ao
        hibrido ARIMA-MLP (mesmo residuo, mas so coincidencia confirmada
        aqui de novo -- o residuo em si independe do estimador a jusante,
        entao bate com ARIMA-MLP por construcao, nao por acaso)."""
        expected_lag_size = {
            "airlines.txt": 20,
            "austres.txt": 1,
            "coloradoRiver.txt": 16,
            "sunspot.txt": 9,
        }
        experiment_params = {
            "linear_model_name": "1arima",
            "diff_kpss": False,
            "horizon": 1,
            "test_size": config.TEST_SIZE,
            "val_size": config.VAL_SIZE,
            "lag_size": "auto",
        }

        for series, expected in expected_lag_size.items():
            _, _, base_info, _ = input_linear_info("chamados", series, experiment_params)
            assert base_info.lag_size_formated == expected, (
                f"{series}: lag_size='auto' (hibrido ARIMA-SVR) resolveu para "
                f"{base_info.lag_size_formated}, esperado {expected}."
            )
