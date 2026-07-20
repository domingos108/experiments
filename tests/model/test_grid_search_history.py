"""
Testes de src/model/grid_search_exp.py -- historico completo do Grid Search
(Tarefa 3.4, PLANO_ARQUITETURA.md Secao 1.6).

`GridSearch._search_params()` ja calculava, por combinacao do grid, a media
das repeticoes internas de erro de validacao -- mas descartava tudo exceto a
combinacao vencedora. Estes testes cobrem a nova chave `grid_search_history`
persistida em CADA entrada de `predict_results`, incluindo uma prova
MECANICA (nao estatistica) de que o historico nunca le `test_metrics`.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import config
from model import generics, grid_search_exp, single_ml_model_exp
from tests.model.conftest import FakeModelExpValOnly, make_fake_grid_search

_make_fake_grid_search = make_fake_grid_search
_FakeModelExpValOnly = FakeModelExpValOnly


class TestGridSearchHistoryMechanics:
    def test_history_has_one_entry_per_grid_combination(self, tmp_path, monkeypatch):
        exec_gs = _make_fake_grid_search(tmp_path, monkeypatch, {"constant": [1.0, 2.0, 3.0]})
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        history = saved[0]["grid_search_history"]

        assert len(history) == 3
        assert {h["params"]["constant"] for h in history} == {1.0, 2.0, 3.0}

    def test_history_never_contains_the_test_metrics_sentinel(self, tmp_path, monkeypatch):
        """Prova mecanica de Zero Data Leakage: se o historico algum dia ler
        test_metrics em vez de val_metrics, o sentinela (999999.0) apareceria
        aqui -- e nao aparece."""
        exec_gs = _make_fake_grid_search(tmp_path, monkeypatch, {"constant": [1.0, 2.0, 3.0]})
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        history = saved[0]["grid_search_history"]

        for entry in history:
            assert entry["val_metric_mean"] != _FakeModelExpValOnly.TEST_SENTINEL
            assert all(v != _FakeModelExpValOnly.TEST_SENTINEL for v in entry["val_metric_reps"])
            assert all(
                v["RMSE"] != _FakeModelExpValOnly.TEST_SENTINEL for v in entry["val_metrics_reps"]
            )

    def test_history_entry_shape_has_params_mean_std_and_reps(self, tmp_path, monkeypatch):
        exec_gs = _make_fake_grid_search(tmp_path, monkeypatch, {"constant": [1.0, 2.0]}, model_exec=3)
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        entry = saved[0]["grid_search_history"][0]

        assert set(entry.keys()) == {
            "params", "val_metric_mean", "val_metric_std", "val_metric_reps", "val_metrics_reps",
        }
        assert len(entry["val_metric_reps"]) == 3  # model_exec repeticoes internas
        assert entry["val_metric_mean"] == pytest.approx(np.mean(entry["val_metric_reps"]))
        assert entry["val_metric_std"] == pytest.approx(np.std(entry["val_metric_reps"]))
        assert len(entry["val_metrics_reps"]) == 3
        assert entry["val_metrics_reps"][0] == {"RMSE": entry["val_metric_reps"][0]}

    def test_history_reflects_the_real_val_metric_value_per_combination(self, tmp_path, monkeypatch):
        """Como o double usa o proprio `constant` como RMSE de validacao (sem
        ruido), o historico deve refletir EXATAMENTE o valor testado -- prova
        de que nao ha mistura entre combinacoes."""
        exec_gs = _make_fake_grid_search(tmp_path, monkeypatch, {"constant": [1.0, 2.0, 3.0]}, model_exec=2)
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        history = saved[0]["grid_search_history"]

        by_constant = {h["params"]["constant"]: h for h in history}
        for constant_value, entry in by_constant.items():
            assert entry["val_metric_mean"] == pytest.approx(constant_value)
            assert entry["val_metric_std"] == pytest.approx(0.0)  # double e deterministico

    def test_history_is_attached_to_every_repetition_not_just_the_first(self, tmp_path, monkeypatch):
        exec_gs = _make_fake_grid_search(tmp_path, monkeypatch, {"constant": [1.0, 2.0]}, model_exec=4)
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)

        assert len(saved) == 4  # model_exec repeticoes finais
        for entry in saved:
            assert "grid_search_history" in entry
            assert len(entry["grid_search_history"]) == 2


class _FakeNonSklearnModelExp:
    """Mimica model_class_exp para wrappers NAO-sklearn (ex. LSTM/NBEATS/ELM
    via neural_forecast_exp.py/perturbative_neural_forecast.py) -- muta
    `experiment_params['model_actual_config']` EM MEMORIA COMPARTILHADA
    dentro de fit_predict(), exatamente como o codigo real desses wrappers
    faz (injeta 'random_seed' e chaves de plumbing no mesmo dict que veio do
    ParameterGrid). Usado para provar que o historico persistido captura os
    hiperparametros do grid, nao o dict ja poluido pela mutacao do wrapper."""

    def __init__(self, model, experiment_id, base_name, model_name, force, normalize, experiment_params):
        self.experiment_params = experiment_params

    def fit_predict(self):
        params = self.experiment_params["model_actual_config"]
        val_rmse = float(params["k"])
        # mutacao in-place do MESMO dict que veio de ParameterGrid -- e o
        # que neural_forecast_exp.py/perturbative_neural_forecast.py fazem
        # de verdade com experiment_params['model_actual_config'].
        params["random_seed"] = 999
        params["injected_plumbing_key"] = "nao_deveria_vazar_no_historico"
        self.metrics_results = {"val_metrics": {"RMSE": val_rmse}}


class TestGridSearchHistoryParamsNotPollutedByNonSklearnMutation:
    """Achado real de code-review (angulo line-by-line): para model_class_exp
    NAO-sklearn, `experiment_params['model_actual_config']` e o MESMO objeto
    dict que veio de ParameterGrid -- se o wrapper mutar esse dict em
    fit_predict() (como neural_forecast_exp.py/perturbative_neural_forecast.py
    fazem de verdade), um snapshot tardio de `params` capturaria a poluicao
    (random_seed nao-reprodutivel, chaves de plumbing) em vez dos
    hiperparametros reais do grid."""

    def test_history_params_reflect_the_grid_not_the_mutated_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        exec_gs = grid_search_exp.GridSearch(
            _FakeNonSklearnModelExp,
            object(),  # nao-sklearn -- is_not_sklearn(model) == True
            {"k": [1, 2]},
            "fake_experiment",
            "airlines.txt",
            "testmodel",
            force=True,
            normalize=True,
            experiment_params={"horizon": 1, "diff_kpss": False},
            model_exec=2,
        )
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        history = saved[0]["grid_search_history"]

        assert len(history) == 2
        for entry in history:
            assert set(entry["params"].keys()) == {"k"}
            assert "random_seed" not in entry["params"]
            assert "injected_plumbing_key" not in entry["params"]


class TestGridSearchHistoryOptOut:
    def test_save_grid_history_false_omits_the_key_entirely(self, tmp_path, monkeypatch):
        exec_gs = _make_fake_grid_search(
            tmp_path, monkeypatch, {"constant": [1.0, 2.0]}, save_grid_history=False
        )
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)

        for entry in saved:
            assert "grid_search_history" not in entry

    def test_save_grid_history_defaults_to_true(self, tmp_path, monkeypatch):
        exec_gs = _make_fake_grid_search(tmp_path, monkeypatch, {"constant": [1.0]})
        assert exec_gs.save_grid_history is True


class TestGridSearchHistoryRealIntegration:
    """Um teste end-to-end com o pipeline real (SKlearnModel + LinearRegression,
    sem double) -- garante que a captura funciona atraves do fluxo de
    producao de verdade, nao so contra o test double acima."""

    def test_history_captured_through_real_sklearn_model_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        exec_gs = grid_search_exp.GridSearch(
            single_ml_model_exp.SKlearnModel,
            LinearRegression(),
            {"fit_intercept": [True, False]},
            "fake_experiment",
            "airlines.txt",
            "testmodel",
            force=True,
            normalize=True,
            experiment_params={"horizon": 1, "diff_kpss": False},
            model_exec=2,
        )
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        history = saved[0]["grid_search_history"]

        assert len(history) == 2
        assert {h["params"]["fit_intercept"] for h in history} == {True, False}
        for entry in history:
            assert np.isfinite(entry["val_metric_mean"])
