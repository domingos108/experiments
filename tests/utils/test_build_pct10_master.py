"""
Testes de src/utils/build_pct10_master.py.

Nao reimplementa nada de build_benchmark_master.py -- so confirma que o
mapeamento PCT10_FAMILIES (nomes de modelo/diretorio da rodada pct10) produz,
via build_master() ja testado, o mesmo contrato de tabela LONG da matriz
'auto' (COLUMN_ORDER), com ARIMA de fora e o filtro `linear_model_name_to_exclude`
aplicado nas 2 familias hibridas. Mesmo padrao de fixture (.pkl sinteticos via
generics.save_result/format_names) de tests/utils/test_build_benchmark_master.py.
"""

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import config
from model import generics
from model.feature_selection import TimeSeriesFeatureSelector
from utils.build_benchmark_master import COLUMN_ORDER, build_master
from utils.build_pct10_master import PCT10_FAMILIES


def _metrics(rmse=1.0):
    return {"test_metrics": {"MSE": rmse, "RMSE": rmse, "MAE": rmse, "MAPE": rmse,
                              "theil": rmse, "ARV": rmse, "IA": rmse, "POCID": 50.0}}


class _FakeBaselineExperiment:
    def __init__(self, rmse=1.0):
        self.model = MLPRegressor(max_iter=5)
        self.metrics_results = _metrics(rmse)


class _FakeFsExperiment:
    def __init__(self, rmse=1.0, n_features=6, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.normal(size=(30, n_features))
        y = 5.0 * X[:, 0] - 3.0 * X[:, 2] + 0.01 * rng.normal(size=30)
        self.model = Pipeline([
            ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
            ("estimator", MLPRegressor(max_iter=5, random_state=seed)),
        ]).fit(X, y)
        self.metrics_results = _metrics(rmse)


def _save_baseline(experiment_id, base_name, model_name, rmse=1.0):
    fold, title = generics.format_names(experiment_id, base_name, model_name)
    generics.save_result(fold, title, [{"experiment": _FakeBaselineExperiment(rmse), "val_metric": 1.0}])


def _save_fs_variant(experiment_id, base_name, model_name, rmse=1.0, n_features=6):
    fold, title = generics.format_names(experiment_id, base_name, model_name)
    generics.save_result(
        fold, title,
        [{"experiment": _FakeFsExperiment(rmse=rmse, n_features=n_features), "val_metric": 1.0}],
    )


class TestPct10FamiliesMapping:
    def test_four_families_no_arima(self):
        names = {f["name"] for f in PCT10_FAMILIES}
        assert names == {"MLP", "SVR", "ARIMA-MLP", "ARIMA-SVR"}

    def test_baseline_model_names_carry_pct10_suffix(self):
        by_name = {f["name"]: f for f in PCT10_FAMILIES}
        assert by_name["MLP"]["baseline_model_name"] == "1mlppct10"
        assert by_name["SVR"]["baseline_model_name"] == "1svrpct10"
        assert by_name["ARIMA-MLP"]["baseline_model_name"] == "1amv1pct10"
        assert by_name["ARIMA-SVR"]["baseline_model_name"] == "1aspct10"

    def test_hybrid_families_exclude_the_copied_arima_model(self):
        by_name = {f["name"]: f for f in PCT10_FAMILIES}
        assert by_name["ARIMA-MLP"]["linear_model_name_to_exclude"] == "1arima"
        assert by_name["ARIMA-SVR"]["linear_model_name_to_exclude"] == "1arima"
        assert by_name["MLP"]["linear_model_name_to_exclude"] is None
        assert by_name["SVR"]["linear_model_name_to_exclude"] is None


class TestBuildMasterWithPct10Families:
    def test_synthetic_run_matches_auto_column_contract_and_excludes_copied_arima(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        _save_baseline("chamados_pct10", "airlines.txt", "1amv1pct10", rmse=18.719077)
        _save_baseline("chamados_pct10_fs_ftest", "airlines.txt", "1arima", rmse=99.0)  # copiado, deve ser filtrado
        _save_fs_variant("chamados_pct10_fs_ftest", "airlines.txt", "1amv1pct10ftest", rmse=21.964239, n_features=13)

        families = [f for f in PCT10_FAMILIES if f["name"] == "ARIMA-MLP"]
        df = build_master(tmp_path / "chamados_pct10", tmp_path, families=families, series_filter=["airlines"])

        assert list(df.columns) == COLUMN_ORDER
        assert set(df["Metodo_FS"]) == {"sem_FS", "ftest"}
        ftest_row = df[df["Metodo_FS"] == "ftest"].iloc[0]
        assert ftest_row["RMSE_FS"] == pytest.approx(21.964239)  # nao 99.0 (o ARIMA copiado)
        assert ftest_row["PctGain"] == pytest.approx((18.719077 - 21.964239) / 18.719077 * 100)
