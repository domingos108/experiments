"""
Testes de src/utils/build_benchmark_master.py.

Formaliza como script versionado a geracao de
results/benchmark_master_with_sources_v1.csv (ate agora produzida por um
script Python executado fora do repositorio, sem rastreabilidade). Reusa
compare_fs_vs_baseline.build_comparison() por familia (uma chamada por
metrica em METRIC_KEYS, ja que build_comparison() so expoe uma metrica por
vez) e consolida em formato LONG (uma linha por Familia x Serie x
Metodo_FS), com as colunas de rastreamento de origem ja presentes no CSV
existente.

Nenhum teste aqui roda um notebook real ou toca data/result/ de verdade --
constroi .pkl sinteticos via generics.save_result/format_names, mesmo padrao
de tests/utils/test_compare_fs_vs_baseline.py.
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
from utils.build_benchmark_master import (
    COLUMN_ORDER,
    FS_DEV_SERIES,
    STRATEGIES,
    TRIVIAL_SERIES,
    build_family_rows,
    build_master,
    fs_dirs_for_family,
)


def _metrics(mse=1.0, rmse=1.0, mae=1.0, mape=1.0, theil=1.0, arv=1.0, ia=1.0, pocid=50.0):
    return {
        "test_metrics": {
            "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape,
            "theil": theil, "ARV": arv, "IA": ia, "POCID": pocid,
        },
    }


class _FakeBaselineExperiment:
    """Baseline real (Additive/SKlearnModel/Arima): model simples, sem
    Pipeline/seletor -- exatamente o formato salvo em data/result/chamados/."""

    def __init__(self, **metric_kwargs):
        self.model = MLPRegressor(max_iter=5)
        self.metrics_results = _metrics(**metric_kwargs)


class _FakeFsExperiment:
    """Variante FS: model = Pipeline fitted com TimeSeriesFeatureSelector,
    exatamente como GridSearch.execution() salva de verdade."""

    def __init__(self, strategy="f_test", seed=0, n_features=6, **metric_kwargs):
        rng = np.random.RandomState(seed)
        X = rng.normal(size=(30, n_features))
        y = 5.0 * X[:, 0] - 3.0 * X[:, 2] + 0.01 * rng.normal(size=30)
        kwargs = {"strategy": strategy, "random_state": seed}
        if strategy in ("f_test", "mutual_info"):
            kwargs["k"] = 3
        self.model = Pipeline([
            ("selector", TimeSeriesFeatureSelector(**kwargs)),
            ("estimator", MLPRegressor(max_iter=5, random_state=seed)),
        ]).fit(X, y)
        self.metrics_results = _metrics(**metric_kwargs)


def _save_baseline(experiment_id, base_name, model_name, **metric_kwargs):
    fold, title = generics.format_names(experiment_id, base_name, model_name)
    generics.save_result(fold, title, [{"experiment": _FakeBaselineExperiment(**metric_kwargs), "val_metric": 1.0}])


def _save_fs_variant(experiment_id, base_name, model_name, strategy, n_features=6, **metric_kwargs):
    fold, title = generics.format_names(experiment_id, base_name, model_name)
    generics.save_result(
        fold, title,
        [{"experiment": _FakeFsExperiment(strategy=strategy, n_features=n_features, **metric_kwargs), "val_metric": 1.0}],
    )


# Estrategia sklearn interna (f_test/mutual_info/rf_embedded/lasso/rfecv) por
# rotulo de diretorio sem underscore (ftest/mutualinfo/rfembedded/lasso/rfecv)
# -- mesma convencao ja usada pelos notebooks reais (Tarefa 5, Secao 1.8 do
# PLANO_ARQUITETURA.md).
_STRATEGY_SLUG_TO_INTERNAL = {
    "ftest": "f_test",
    "mutualinfo": "mutual_info",
    "rfembedded": "rf_embedded",
    "lasso": "lasso",
    "rfecv": "rfecv",
}


class TestFsDirsForFamily:
    def test_none_prefix_yields_no_fs_dirs(self, tmp_path):
        assert fs_dirs_for_family(tmp_path, None) == {}

    def test_prefix_expands_to_one_dir_per_strategy(self, tmp_path):
        dirs = fs_dirs_for_family(tmp_path, "chamados_v4_fs_mlp")

        assert set(dirs.keys()) == set(STRATEGIES)
        assert dirs["ftest"] == tmp_path / "chamados_v4_fs_mlp_ftest"
        assert dirs["rfecv"] == tmp_path / "chamados_v4_fs_mlp_rfecv"


class TestBuildFamilyRowsWithoutFeatureSelection:
    """Familia tipo ARIMA: fs_dir_prefix=None -- so a linha sem_FS existe,
    NFeatures fica vazio (nao ha janelamento/FS nessa familia) e PctGain=0."""

    def test_single_series_yields_one_sem_fs_row(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("chamados", "airlines.txt", "1arima", rmse=19.728002, mse=389.194049)

        family = {
            "name": "ARIMA",
            "baseline_model_name": "1arima",
            "linear_model_name_to_exclude": None,
            "fs_dir_prefix": None,
        }
        rows = build_family_rows(tmp_path / "chamados", tmp_path, family)

        assert len(rows) == 1
        row = rows[0]
        assert row["Familia"] == "ARIMA"
        assert row["Serie"] == "airlines"
        assert row["Metodo_FS"] == "sem_FS"
        assert row["PctGain"] == pytest.approx(0.0)
        assert row["RMSE_Baseline"] == pytest.approx(19.728002)
        assert row["RMSE_FS"] == pytest.approx(19.728002)
        assert row["MSE_Baseline"] == pytest.approx(389.194049)
        assert np.isnan(row["NFeatures"])
        assert row["Pkl_Source_Dir"] == "data/result/chamados/"
        assert row["Metrics_Source_File"] == "results/baseline_metrics.csv"
        assert row["Features_Source_File"] == ""


class TestBuildFamilyRowsWithFeatureSelection:
    def _setup(self, tmp_path, monkeypatch, base_name="airlines.txt", serie="airlines"):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline(
            "chamados", base_name, "1mlp",
            rmse=27.697682, mse=785.157998, mae=23.468313, mape=5.109149,
            theil=0.418761, arv=0.127838, ia=0.96615, pocid=80.714286,
        )
        _save_fs_variant(
            "chamados_v4_fs_mlp_ftest", base_name, "1mlpftest", "f_test", n_features=20,
            rmse=21.613648, mse=470.771485, mae=17.215984, mape=3.747392,
            theil=0.217904, arv=0.086003, ia=0.978797, pocid=82.857143,
        )

    def test_produces_sem_fs_row_plus_one_row_per_strategy_present(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        family = {
            "name": "MLP",
            "baseline_model_name": "1mlp",
            "linear_model_name_to_exclude": None,
            "fs_dir_prefix": "chamados_v4_fs_mlp",
        }

        rows = build_family_rows(tmp_path / "chamados", tmp_path, family)

        methods = [r["Metodo_FS"] for r in rows]
        # so ftest foi de fato gerado nesta fixture -- as outras 4 pastas nao
        # existem, entao build_comparison() nao pode inventar uma linha para
        # elas (mesmo invariante de "missing series vira NaN, nao crash" do
        # compare_fs_vs_baseline.py original).
        assert methods == ["sem_FS", "ftest"]

    def test_pct_gain_matches_rmse_based_formula(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        family = {
            "name": "MLP",
            "baseline_model_name": "1mlp",
            "linear_model_name_to_exclude": None,
            "fs_dir_prefix": "chamados_v4_fs_mlp",
        }

        rows = build_family_rows(tmp_path / "chamados", tmp_path, family)
        ftest_row = next(r for r in rows if r["Metodo_FS"] == "ftest")

        expected = (27.697682 - 21.613648) / 27.697682 * 100
        assert ftest_row["PctGain"] == pytest.approx(expected)

    def test_all_metric_columns_populated_not_only_rmse(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        family = {
            "name": "MLP",
            "baseline_model_name": "1mlp",
            "linear_model_name_to_exclude": None,
            "fs_dir_prefix": "chamados_v4_fs_mlp",
        }

        rows = build_family_rows(tmp_path / "chamados", tmp_path, family)
        ftest_row = next(r for r in rows if r["Metodo_FS"] == "ftest")
        sem_fs_row = rows[0]

        assert ftest_row["MSE_Baseline"] == pytest.approx(785.157998)
        assert ftest_row["MSE_FS"] == pytest.approx(470.771485)
        assert ftest_row["MAE_FS"] == pytest.approx(17.215984)
        assert ftest_row["MAPE_FS"] == pytest.approx(3.747392)
        assert ftest_row["theil_FS"] == pytest.approx(0.217904)
        assert ftest_row["ARV_FS"] == pytest.approx(0.086003)
        assert ftest_row["IA_FS"] == pytest.approx(0.978797)
        assert ftest_row["POCID_FS"] == pytest.approx(82.857143)
        # sem_FS: Baseline == FS em toda metrica (comparando o baseline com
        # ele mesmo), nao so em RMSE.
        assert sem_fs_row["MAE_Baseline"] == sem_fs_row["MAE_FS"]
        assert sem_fs_row["POCID_Baseline"] == sem_fs_row["POCID_FS"]

    def test_nfeatures_for_baseline_row_comes_from_fs_variant_n_features_total(self, tmp_path, monkeypatch):
        """NFeatures da linha sem_FS nao existe em nenhum .pkl de baseline
        (nao ha selector) -- vem de selector.n_features_in_ de QUALQUER
        variante FS da mesma familia/serie, que e constante entre estrategias
        (mesmo lag_size='auto' resolvido, PLANO_ARQUITETURA.md Secao 1.8)."""
        self._setup(tmp_path, monkeypatch)
        family = {
            "name": "MLP",
            "baseline_model_name": "1mlp",
            "linear_model_name_to_exclude": None,
            "fs_dir_prefix": "chamados_v4_fs_mlp",
        }

        rows = build_family_rows(tmp_path / "chamados", tmp_path, family)
        sem_fs_row = rows[0]

        assert sem_fs_row["NFeatures"] == pytest.approx(20)

    def test_features_and_metrics_source_file_columns_point_to_the_fs_dir(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        family = {
            "name": "MLP",
            "baseline_model_name": "1mlp",
            "linear_model_name_to_exclude": None,
            "fs_dir_prefix": "chamados_v4_fs_mlp",
        }

        rows = build_family_rows(tmp_path / "chamados", tmp_path, family)
        ftest_row = next(r for r in rows if r["Metodo_FS"] == "ftest")

        assert ftest_row["Metrics_Source_File"] == "results/chamados_v4_fs_mlp_ftest/metrics.csv"
        assert ftest_row["Pkl_Source_Dir"] == "data/result/chamados_v4_fs_mlp_ftest/"
        assert ftest_row["Features_Source_File"] == "results/chamados_v4_fs_mlp_ftest/selected_features.csv"


class TestTrivialFlag:
    def test_austres_is_flagged_trivial_others_are_not(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("chamados", "airlines.txt", "1arima", rmse=1.0)
        _save_baseline("chamados", "austres.txt", "1arima", rmse=1.0)

        family = {
            "name": "ARIMA",
            "baseline_model_name": "1arima",
            "linear_model_name_to_exclude": None,
            "fs_dir_prefix": None,
        }
        rows = build_family_rows(tmp_path / "chamados", tmp_path, family)

        by_serie = {r["Serie"]: r["Trivial"] for r in rows}
        assert by_serie["airlines"] is False
        assert by_serie["austres"] is True
        assert TRIVIAL_SERIES == {"austres"}


class TestSeriesFilter:
    """data/result/chamados/ real tem baselines de 17 series, mas so as 4 de
    FS_DEV_SERIES tem pastas chamados_v4_fs_* -- sem filtro, a linha sem_FS
    vazaria escopo para as outras 13 (RUNBOOK.md Secao 1: expandir alem das 4
    e decisao explicita futura, nao implicita)."""

    def test_none_filter_keeps_every_series_found_in_baseline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("chamados", "airlines.txt", "1arima", rmse=1.0)
        _save_baseline("chamados", "milk.txt", "1arima", rmse=1.0)

        family = {"name": "ARIMA", "baseline_model_name": "1arima", "linear_model_name_to_exclude": None, "fs_dir_prefix": None}
        rows = build_family_rows(tmp_path / "chamados", tmp_path, family, series_filter=None)

        assert {r["Serie"] for r in rows} == {"airlines", "milk"}

    def test_explicit_filter_excludes_series_not_in_the_allowlist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("chamados", "airlines.txt", "1arima", rmse=1.0)
        _save_baseline("chamados", "milk.txt", "1arima", rmse=1.0)

        family = {"name": "ARIMA", "baseline_model_name": "1arima", "linear_model_name_to_exclude": None, "fs_dir_prefix": None}
        rows = build_family_rows(tmp_path / "chamados", tmp_path, family, series_filter={"airlines"})

        assert {r["Serie"] for r in rows} == {"airlines"}

    def test_build_master_default_series_filter_argument_is_respected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("chamados", "airlines.txt", "1arima", rmse=1.0)
        _save_baseline("chamados", "milk.txt", "1arima", rmse=1.0)

        families = [{"name": "ARIMA", "baseline_model_name": "1arima", "linear_model_name_to_exclude": None, "fs_dir_prefix": None}]
        df = build_master(tmp_path / "chamados", tmp_path, families=families, series_filter=FS_DEV_SERIES)

        assert list(df["Serie"]) == ["airlines"]


class TestBuildMaster:
    def test_concatenates_rows_from_multiple_families_with_expected_column_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("chamados", "airlines.txt", "1arima", rmse=19.728002)
        _save_baseline("chamados", "airlines.txt", "1mlp", rmse=27.697682)
        _save_fs_variant("chamados_v4_fs_mlp_ftest", "airlines.txt", "1mlpftest", "f_test", rmse=21.613648)

        families = [
            {"name": "ARIMA", "baseline_model_name": "1arima", "linear_model_name_to_exclude": None, "fs_dir_prefix": None},
            {"name": "MLP", "baseline_model_name": "1mlp", "linear_model_name_to_exclude": None, "fs_dir_prefix": "chamados_v4_fs_mlp"},
        ]

        df = build_master(tmp_path / "chamados", tmp_path, families=families)

        assert list(df.columns) == COLUMN_ORDER
        assert set(df["Familia"]) == {"ARIMA", "MLP"}
        assert len(df) == 3  # ARIMA/sem_FS + MLP/sem_FS + MLP/ftest

    def test_linear_model_name_to_exclude_filters_copied_arima_from_hybrid_fs_dirs(self, tmp_path, monkeypatch):
        """RUNBOOK.md manda copiar <serie>_1arima.pkl para dentro de cada
        pasta chamados_v4_fs_* das familias hibridas -- sem o filtro, a
        comparacao contaria o ARIMA copiado como se fosse a variante FS."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("chamados", "airlines.txt", "1amv1", rmse=17.43618)
        _save_baseline("chamados_v4_fs_ftest", "airlines.txt", "1arima", rmse=99.0)  # copiado
        _save_fs_variant("chamados_v4_fs_ftest", "airlines.txt", "1amv1ftest", "f_test", rmse=19.113854)

        family = {
            "name": "ARIMA-MLP",
            "baseline_model_name": "1amv1",
            "linear_model_name_to_exclude": "1arima",
            "fs_dir_prefix": "chamados_v4_fs",
        }
        rows = build_family_rows(tmp_path / "chamados", tmp_path, family)
        ftest_row = next(r for r in rows if r["Metodo_FS"] == "ftest")

        assert ftest_row["RMSE_FS"] == pytest.approx(19.113854)  # nao 99.0
