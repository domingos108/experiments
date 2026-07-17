"""
Testes de src/utils/export_metrics_to_csv.py::run_export_metrics_to_csv
(Tarefa 3.2): nucleo reutilizavel extraido de main() para ser chamado tanto
pela CLI quanto direto de um notebook, sem sys.exit() -- ver
PLANO_ARQUITETURA.md/RUNBOOK.md, fluxo notebook-only.

Reusa o padrao de _build_fake_pkl_result ja usado em
tests/metrics/test_metrics.py -- .pkl real via generics.save_result, nunca
mock do pickle em si.
"""

from pathlib import Path

import pytest

import config
from model import generics
from utils.export_metrics_to_csv import run_export_metrics_to_csv


def _build_fake_pkl_result(rmse=1.0):
    metrics_results = {
        "test_metrics": {
            "RMSE": rmse, "MSE": rmse ** 2, "MAE": rmse, "MAPE": rmse,
            "theil": 1.0, "ARV": 1.0, "IA": 1.0, "POCID": 50.0,
        },
        "time_exec": {"training": 0.1, "testing": 0.01},
    }
    return [{"experiment": generics.ResultExp(metrics_results), "val_metric": rmse}]


class TestRunExportMetricsToCsv:
    def test_raises_filenotfound_when_result_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_export_metrics_to_csv(tmp_path / "does_not_exist", tmp_path / "out.csv")

    def test_writes_csv_and_returns_populated_dataframe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        fold, title = generics.format_names("fake_experiment", "airlines.txt", "1amv1")
        generics.save_result(fold, title, _build_fake_pkl_result(rmse=3.0))

        output = tmp_path / "metrics.csv"
        df = run_export_metrics_to_csv(tmp_path / "fake_experiment", output, detail=True)

        assert not df.empty
        assert output.exists()
        assert output.with_name("metrics_detail.csv").exists()
        assert df.loc[df["Serie"] == "airlines", "RMSE_mean"].iloc[0] == pytest.approx(3.0)

    def test_returns_empty_dataframe_without_writing_when_no_pkl_found(self, tmp_path):
        empty_dir = tmp_path / "empty_experiment"
        empty_dir.mkdir()

        output = tmp_path / "metrics.csv"
        df = run_export_metrics_to_csv(empty_dir, output)

        assert df.empty
        assert not output.exists()
