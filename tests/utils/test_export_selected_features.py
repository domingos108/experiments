"""
Testes de src/utils/export_selected_features.py (Tarefa 3.1, Parte C).

Decisao de design (PLANO_ARQUITETURA.md Secao 1.5, validada por brainstorming
antes de implementar): o Pipeline fitted (com o seletor ja ajustado) ja
sobrevive dentro do .pkl existente, sem qualquer mudanca em generics.py --
este script e uma extracao POS-HOC, no mesmo padrao de
src/utils/export_metrics_to_csv.py. Os testes aqui nunca tocam data/result/
real -- usam tmp_path + monkeypatch(config.MODEL_DATA_PATH), montando um
Pipeline real e fitted (nao mock) via generics.save_result/format_names,
exatamente como GridSearch.execution() salva de verdade.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import config
from model import generics
from model.feature_selection import TimeSeriesFeatureSelector
from utils.export_selected_features import collect_all_rows, extract_rows, run_export_selected_features


def _fit_pipeline(strategy, n_features=6, n_samples=40, k=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = 5.0 * X[:, 0] - 3.0 * X[:, 2] + 0.01 * rng.normal(size=n_samples)

    kwargs = {"strategy": strategy, "random_state": seed}
    if strategy in ("f_test", "mutual_info"):
        kwargs["k"] = k

    pipeline = Pipeline([
        ("selector", TimeSeriesFeatureSelector(**kwargs)),
        ("estimator", MLPRegressor(max_iter=200, random_state=seed)),
    ])
    pipeline.fit(X, y)
    return pipeline


class _FakeExperiment:
    """Mimica Additive/SKlearnModel apos fit_predict(): self.model e o
    Pipeline ja ajustado (mutado in place por fit_predict_ml_schemma)."""

    def __init__(self, model):
        self.model = model
        self.metrics_results = {"dummy": True}


class _FakeBaselineExperiment:
    """Mimica um baseline (SKlearnModel com MLPRegressor puro, sem Pipeline/
    seletor) -- precisa ser modulo-level para ser picklable, igual a
    generics.save_result exige de verdade."""

    def __init__(self):
        rng = np.random.RandomState(0)
        self.model = MLPRegressor(max_iter=10).fit(rng.normal(size=(20, 3)), rng.normal(size=20))
        self.metrics_results = {"dummy": True}


def _save_fake_result(base_name, model_name, strategy, n_repeticoes=1, **fit_kwargs):
    fold, title = generics.format_names("fake_fs_experiment", base_name, model_name)
    predict_results = []
    for rep in range(n_repeticoes):
        pipeline = _fit_pipeline(strategy, seed=rep, **fit_kwargs)
        predict_results.append({"experiment": _FakeExperiment(pipeline), "val_metric": 0.1})
    generics.save_result(fold, title, predict_results)
    return Path(title)


class TestExtractRowsFromPklWithSelector:
    def test_extracts_one_row_per_repetition_with_correct_strategy_and_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        title = _save_fake_result("sunspot.txt", "1amv1rfembedded", "rf_embedded", n_repeticoes=3)

        rows = extract_rows(title)

        assert len(rows) == 3
        for i, row in enumerate(rows):
            assert row["ExperimentID"] == "fake_fs_experiment"
            assert row["Serie"] == "sunspot"
            assert row["Modelo"] == "1amv1rfembedded"
            assert row["Repeticao"] == i
            assert row["Strategy"] == "rf_embedded"
            assert row["N_Features_Total"] == 6
            assert 1 <= row["N_Features_Selected"] <= 6

    def test_k_based_strategy_selects_exactly_k_features(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        title = _save_fake_result("airlines.txt", "1amv1ftest", "f_test", n_repeticoes=1, k=3)

        rows = extract_rows(title)

        assert rows[0]["N_Features_Selected"] == 3

    def test_selected_indices_and_lag_names_are_consistent(self, tmp_path, monkeypatch):
        """Coluna j (0-indexado) do X de treino corresponde a lag_{n_total - j}
        (create_windowing/input.py nomeia lag_L..lag_1 -- ver PLANO_ARQUITETURA.md
        Secao 1.5). Prova mecanica do mapeamento, nao um valor fixo hardcoded."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        title = _save_fake_result("austres.txt", "1amv1lasso", "lasso", n_repeticoes=1)

        row = extract_rows(title)[0]
        indices = [int(i) for i in row["Selected_Indices"].split(";")]
        lag_names = row["Selected_Lag_Names"].split(";")
        n_total = row["N_Features_Total"]

        assert lag_names == [f"lag_{n_total - idx}" for idx in indices]


class TestExtractRowsSkipsPklWithoutSelector:
    def test_baseline_pkl_without_selector_is_skipped_not_errored(self, tmp_path, monkeypatch):
        """.pkl dos 5 baselines (MLPRegressor puro, sem Pipeline/seletor) --
        precisa ser ignorado silenciosamente (aviso, nao excecao), ja que a
        Tarefa 3.1 e estritamente aditiva e nao pode quebrar leitura de
        experiment_ids antigos."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        fold, title = generics.format_names("fake_baseline_experiment", "airlines.txt", "1mlp")

        generics.save_result(fold, title, [{"experiment": _FakeBaselineExperiment(), "val_metric": 0.1}])

        rows = extract_rows(Path(title))

        assert rows == []

    def test_pkl_from_pre_tarefa_3_1_selector_without_n_features_in_is_skipped_not_errored(self, tmp_path, monkeypatch, capsys):
        """Achado real de code-review: data/result/chamados_v2_fs_ftest/ (Tarefa 2,
        ja executado de verdade) tem selectors fitted pela versao ANTIGA de
        TimeSeriesFeatureSelector, que nunca setava n_features_in_ (atributo
        novo desta sessao). extract_rows precisa pular essa linha graciosamente
        (mesmo caminho de '.pkl sem seletor'), nao cair na excecao generica com
        um aviso enganoso de 'Erro na repeticao' -- verificado via a mensagem
        impressa, ja que ambos os caminhos produzem `rows == []`."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        fold, title = generics.format_names("fake_pre_tarefa31_experiment", "airlines.txt", "1amv1ftest")

        pipeline = _fit_pipeline("f_test", k=3)
        del pipeline.named_steps["selector"].n_features_in_  # simula selector pre-Tarefa-3.1

        generics.save_result(fold, title, [{"experiment": _FakeExperiment(pipeline), "val_metric": 0.1}])

        rows = extract_rows(Path(title))
        captured = capsys.readouterr()

        assert rows == []
        assert "Erro na repetição" not in captured.out


class TestCollectAllRows:
    def test_aggregates_rows_from_multiple_pkl_files_in_result_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_fake_result("sunspot.txt", "1amv1rfembedded", "rf_embedded", n_repeticoes=2)
        _save_fake_result("airlines.txt", "1amv1lasso", "lasso", n_repeticoes=2)

        result_dir = tmp_path / "fake_fs_experiment"
        df = collect_all_rows(result_dir)

        assert len(df) == 4
        assert set(df["Serie"]) == {"sunspot", "airlines"}


class TestRunExportSelectedFeatures:
    """Nucleo reutilizavel extraido de main() na Tarefa 3.2, para chamada
    direta de notebook (sem sys.exit) -- mesma funcao alimenta CLI e
    notebook, garantindo saida identica entre os dois pontos de entrada."""

    def test_raises_filenotfound_when_result_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_export_selected_features(tmp_path / "does_not_exist", tmp_path / "out.csv")

    def test_writes_csv_and_returns_populated_dataframe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_fake_result("sunspot.txt", "1amv1rfembedded", "rf_embedded", n_repeticoes=2)

        output = tmp_path / "selected_features.csv"
        df = run_export_selected_features(tmp_path / "fake_fs_experiment", output, detail=True)

        assert not df.empty
        assert output.exists()
        assert output.with_name("selected_features_detail.csv").exists()

    def test_returns_empty_dataframe_without_writing_when_no_selector_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        fold, title = generics.format_names("fake_baseline_experiment", "airlines.txt", "1mlp")
        generics.save_result(fold, title, [{"experiment": _FakeBaselineExperiment(), "val_metric": 0.1}])

        output = tmp_path / "selected_features.csv"
        df = run_export_selected_features(tmp_path / "fake_baseline_experiment", output)

        assert df.empty
        assert not output.exists()
