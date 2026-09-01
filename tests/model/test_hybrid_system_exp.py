"""
Testes de src/model/hybrid_system_exp.py -- Tarefa 7 do roadmap
(PLANO_ARQUITETURA.md): confirma, com evidencia real (nao suposicao), que a
combinacao Additive + Pipeline([selector, SVR]) funciona sem nenhuma mudanca
de codigo -- fecha a matriz completa de 5 familias (ARIMA, ARIMA-MLP, MLP,
SVR, ARIMA-SVR) x 5 metodos de FS. Additive ja era comprovadamente agnostico
a identidade de `model` com MLPRegressor (Tarefas 2/3); esta tarefa e a
primeira vez que SVR e testado dentro de Additive.

Tarefa 15/16 (spike, PLANO_ARQUITETURA.md Secao 4.1 -- "risco a validar antes
de generalizar a solucao"): estende a mesma pergunta para os wrappers
NAO-aditivos KhasheiBijariHybrid/NoLiCHybrid, que constroem matrizes de
entrada combinadas fora do fluxo padrao. Resultado (evidencia real, nao
suposicao): KhasheiBijariHybrid.fit_predict() so chama
generics.fit_predict_ml_schemma(self.model, ...) -- mesma premissa agnostica
ja provada para Additive/SKlearnModel, Pipeline encaixa sem mudanca.
NoLiCHybrid quebra: o estagio combinador (_grid_search_mlp) faz
GridSearchCV(estimator=clone(self.model), param_grid={'hidden_layer_sizes':
..., 'solver': ...}) com nomes de parametro do MLPRegressor direto, sem
prefixo 'estimator__' -- assume que self.model E um MLPRegressor bruto, nao
composicao-aware. Ver docs/spikes/spike_pipeline_khasheibijari_nolic.md.
"""
import shutil
from pathlib import Path

import config
import pytest
from model import generics, grid_search_exp, hybrid_system_exp
from model.feature_selection import TimeSeriesFeatureSelector
from model.hybrid_system_exp import input_linear_info
from sklearn.neural_network import MLPRegressor
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


def _copy_real_arima_pkl(tmp_path, dest_experiment_id, monkeypatch):
    """Copia airlines_1arima.pkl real (data/result/chamados/) para dentro de
    um experiment_id fake, dentro de um MODEL_DATA_PATH temporario -- mesma
    preparacao usada por TestAdditiveAcceptsPipelineWithSVR, extraida aqui
    para reuso pelos 2 testes do spike Tarefa 15/16 sem duplicar a logica de
    setup. O monkeypatch de config.MODEL_DATA_PATH precisa acontecer ANTES
    de copy_pretrained_linear_model (que le config.MODEL_DATA_PATH
    internamente) -- na ordem errada, o .pkl fake vaza para o
    data/result/ real (achado ao rodar este teste pela primeira vez)."""
    real_model_data_path = config.MODEL_DATA_PATH
    tmp_chamados = tmp_path / "chamados"
    tmp_chamados.mkdir()
    shutil.copy(Path(real_model_data_path) / "chamados" / "airlines_1arima.pkl", tmp_chamados)

    monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

    from utils.copy_pretrained_linear_model import copy_pretrained_linear_model

    copy_pretrained_linear_model(
        source_experiment_id="chamados",
        dest_experiment_id=dest_experiment_id,
        series_list=["airlines.txt"],
        linear_model_name="1arima",
    )


class TestKhasheiBijariHybridAcceptsPipeline:
    """Spike Tarefa 15/16: KhasheiBijariHybrid.fit_predict() constroi a
    matriz combinada (residuo + previsao linear + serie) e so chama
    generics.fit_predict_ml_schemma(self.model, ...) -- a mesma funcao
    100% agnostica (model.fit/model.predict) ja usada por Additive/
    SKlearnModel/NonLinear. Confirmado com evidencia real (nao so ausencia
    de erro): o seletor recebe as 41 colunas da matriz completa
    (n1_lags=20 + linear_forecast_t=1 + m1_lags=20 para airlines no
    contexto hibrido), nao uma fatia pre-filtrada."""

    def test_pipeline_with_selector_runs_end_to_end_through_khashei_bijari(self, tmp_path, monkeypatch):
        _copy_real_arima_pkl(tmp_path, "fake_experiment_khashei", monkeypatch)

        model = Pipeline([
            ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
            ("estimator", MLPRegressor(hidden_layer_sizes=(5,), max_iter=200, random_state=42)),
        ])

        exec_gs = grid_search_exp.GridSearch(
            hybrid_system_exp.KhasheiBijariHybrid,
            model,
            {"estimator__hidden_layer_sizes": [(5,)], "estimator__max_iter": [200]},
            "fake_experiment_khashei",
            "airlines.txt",
            "testmodelkhashei",
            force=True,
            normalize=True,
            experiment_params={"linear_model_name": "1arima", "diff_kpss": False, "horizon": 1},
            model_exec=1,
            use_val_slipt_for_prev=True,
        )
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        fitted_selector = saved[0]["experiment"].model.named_steps["selector"]

        # n1_lags=m1_lags=lag_size_formated=20 (default quando nao
        # especificado em experiment_params) + 1 coluna linear_forecast_t =
        # 41 -- confirma que o seletor viu a matriz combinada INTEIRA, nao
        # so os lags do residuo.
        assert fitted_selector.n_features_in_ == 41
        assert fitted_selector.selected_indices_.shape[0] == 3

        test_metrics = saved[0]["experiment"].metrics_results["test_metrics"]
        assert test_metrics != {}


class TestNoLiCHybridBreaksWithPipeline:
    """Spike Tarefa 15/16: DIFERENTE de KhasheiBijariHybrid, o estagio
    combinador (M_c) de NoLiCHybrid faz um GridSearchCV interno
    (_grid_search_mlp, hybrid_system_exp.py) com
    param_grid={'hidden_layer_sizes': ..., 'solver': ...} -- nomes de
    parametro do MLPRegressor DIRETO, sem prefixo 'estimator__'. Isso
    assume que self.model E um MLPRegressor bruto (usado tambem como base
    clonada do combinador, nao so do estagio de residuo) -- quebra com
    Pipeline. Este teste documenta o comportamento ATUAL (quebrado); se
    _grid_search_mlp for corrigido para ser composicao-aware (Tarefa 16+,
    ver docs/spikes/spike_pipeline_khasheibijari_nolic.md), este teste deve
    ser atualizado para esperar sucesso, nao removido silenciosamente."""

    def test_pipeline_raises_valueerror_in_combiner_grid_search(self, tmp_path, monkeypatch):
        _copy_real_arima_pkl(tmp_path, "fake_experiment_nolic", monkeypatch)

        model = Pipeline([
            ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
            ("estimator", MLPRegressor(hidden_layer_sizes=(5,), max_iter=200, random_state=42)),
        ])

        nolic = hybrid_system_exp.NoLiCHybrid(
            model,
            "fake_experiment_nolic",
            "airlines.txt",
            "testmodelnolic",
            force=True,
            normalize=True,
            experiment_params={
                "linear_model_name": "1arima",
                "diff_kpss": False,
                "horizon": 1,
                "test_size": config.TEST_SIZE,
                "val_size": config.VAL_SIZE,
                "lag_size": "auto",
            },
        )

        with pytest.raises(ValueError, match=r"Invalid parameter 'hidden_layer_sizes' for estimator Pipeline"):
            nolic.fit_predict()
