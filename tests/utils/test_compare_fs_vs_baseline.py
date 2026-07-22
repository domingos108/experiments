"""
Testes de src/utils/compare_fs_vs_baseline.py (Tarefa 3.1, Parte D).

Compara o baseline (data/result/chamados/, modelo '1amv1') contra cada
variante de Feature Selection lado a lado, por serie, incluindo o ganho
percentual de RMSE e (quando disponivel) o nº medio de features selecionadas
(reaproveita export_selected_features -- mesmos .pkl, extracao diferente).

Nenhum teste aqui roda um notebook real ou toca data/result/ de verdade --
constroi .pkl sinteticos via generics.save_result/format_names, exatamente
como os scripts de producao leem, seguindo a mesma regra dos testes de
export_metrics_to_csv.py/export_selected_features.py.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import config
from model import generics
from model.feature_selection import TimeSeriesFeatureSelector
from utils.compare_fs_vs_baseline import build_comparison


def _fake_metrics_results(rmse):
    return {
        "test_metrics": {
            "RMSE": rmse, "MSE": rmse ** 2, "MAE": rmse, "MAPE": rmse,
            "theil": 1.0, "ARV": 1.0, "IA": 1.0, "POCID": 50.0,
        },
    }


class _FakeBaselineExperiment:
    """Baseline ARIMA-MLP (Additive) real: model_name '1amv1', sem Pipeline/
    seletor -- exatamente o formato salvo em data/result/chamados/."""

    def __init__(self, rmse):
        self.model = MLPRegressor(max_iter=5)
        self.metrics_results = _fake_metrics_results(rmse)


class _FakeFsExperiment:
    """Variante FS: model = Pipeline fitted com TimeSeriesFeatureSelector,
    exatamente como GridSearch.execution() salva de verdade."""

    def __init__(self, rmse, strategy, seed=0, n_features=6):
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
        self.metrics_results = _fake_metrics_results(rmse)


def _save_baseline(base_name, rmse):
    fold, title = generics.format_names("chamados", base_name, "1amv1")
    generics.save_result(fold, title, [{"experiment": _FakeBaselineExperiment(rmse), "val_metric": rmse}])


def _save_fs_variant(experiment_id, base_name, model_name, strategy, rmse):
    fold, title = generics.format_names(experiment_id, base_name, model_name)
    generics.save_result(fold, title, [{"experiment": _FakeFsExperiment(rmse, strategy), "val_metric": rmse}])


def _save_arima(experiment_id, base_name, rmse):
    """RUNBOOK.md Secao 3/8b manda copiar <serie>_1arima.pkl para DENTRO de
    cada pasta chamados_v4_fs_* (Additive precisa do ARIMA pre-treinado sob o
    mesmo experiment_id). Isso significa que um result_dir de FS real sempre
    tem 2 modelos por serie: '1arima' e a variante FS -- nao so 1."""
    fold, title = generics.format_names(experiment_id, base_name, "1arima")
    generics.save_result(fold, title, [{"experiment": _FakeBaselineExperiment(rmse), "val_metric": rmse}])


class TestBuildComparison:
    def test_computes_pct_gain_for_a_single_strategy_and_series(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("airlines.txt", rmse=10.0)
        _save_fs_variant("chamados_v4_fs_ftest", "airlines.txt", "1amv1ftest", "f_test", rmse=8.0)

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={"ftest": tmp_path / "chamados_v4_fs_ftest"},
        )

        row = df[df["Serie"] == "airlines"].iloc[0]
        assert row["Baseline_RMSE"] == pytest.approx(10.0)
        assert row["ftest_RMSE"] == pytest.approx(8.0)
        assert row["ftest_PctGain"] == pytest.approx(20.0)  # (10-8)/10 * 100

    def test_negative_pct_gain_when_fs_variant_is_worse_than_baseline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("sunspot.txt", rmse=5.0)
        _save_fs_variant("chamados_v4_fs_lasso", "sunspot.txt", "1amv1lasso", "lasso", rmse=6.0)

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={"lasso": tmp_path / "chamados_v4_fs_lasso"},
        )

        row = df[df["Serie"] == "sunspot"].iloc[0]
        assert row["lasso_PctGain"] == pytest.approx(-20.0)  # (5-6)/5 * 100

    def test_side_by_side_columns_for_multiple_strategies(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("austres.txt", rmse=4.0)
        _save_fs_variant("chamados_v4_fs_ftest", "austres.txt", "1amv1ftest", "f_test", rmse=3.0)
        _save_fs_variant("chamados_v4_fs_rfembedded", "austres.txt", "1amv1rfembedded", "rf_embedded", rmse=3.5)

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={
                "ftest": tmp_path / "chamados_v4_fs_ftest",
                "rfembedded": tmp_path / "chamados_v4_fs_rfembedded",
            },
        )

        row = df[df["Serie"] == "austres"].iloc[0]
        assert {"ftest_RMSE", "ftest_PctGain", "rfembedded_RMSE", "rfembedded_PctGain"}.issubset(df.columns)
        assert row["ftest_RMSE"] == pytest.approx(3.0)
        assert row["rfembedded_RMSE"] == pytest.approx(3.5)

    def test_includes_mean_n_features_selected_column_when_available(self, tmp_path, monkeypatch):
        """Requisito da Parte D, item 2: incluir o nº de features selecionadas
        ao lado de cada variante FS -- reaproveita export_selected_features
        sobre os MESMOS .pkl (nao uma segunda fonte de dados)."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("coloradoRiver.txt", rmse=2.0)
        _save_fs_variant("chamados_v4_fs_rfembedded", "coloradoRiver.txt", "1amv1rfembedded", "rf_embedded", rmse=1.5)

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={"rfembedded": tmp_path / "chamados_v4_fs_rfembedded"},
        )

        row = df[df["Serie"] == "coloradoRiver"].iloc[0]
        assert "rfembedded_NFeatures" in df.columns
        assert 1 <= row["rfembedded_NFeatures"] <= 6

    def test_missing_series_in_fs_variant_yields_nan_not_crash(self, tmp_path, monkeypatch):
        """Uma serie que ainda nao foi rodada para uma variante especifica
        nao pode derrubar a comparacao inteira -- vira NaN, visivel no CSV."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("airlines.txt", rmse=10.0)
        _save_baseline("sunspot.txt", rmse=5.0)
        _save_fs_variant("chamados_v4_fs_ftest", "airlines.txt", "1amv1ftest", "f_test", rmse=8.0)
        # sunspot.txt nao tem variante ftest ainda

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={"ftest": tmp_path / "chamados_v4_fs_ftest"},
        )

        row = df[df["Serie"] == "sunspot"].iloc[0]
        assert np.isnan(row["ftest_RMSE"])
        assert np.isnan(row["ftest_PctGain"])

    def test_ignores_the_copied_arima_pkl_that_lives_inside_every_fs_result_dir(self, tmp_path, monkeypatch):
        """Achado real de code-review: RUNBOOK.md manda copiar <serie>_1arima.pkl
        para dentro de CADA pasta chamados_v4_fs_* (Additive depende do ARIMA
        pre-treinado sob o mesmo experiment_id). Sem filtrar por Modelo, um
        result_dir de FS real tem 2 linhas por serie (1arima + variante FS),
        o que faz .set_index('Serie') gerar indice duplicado e
        df['Serie'].map(...) explodir com pandas.errors.InvalidIndexError."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("airlines.txt", rmse=10.0)
        _save_arima("chamados_v4_fs_ftest", "airlines.txt", rmse=12.0)  # copiado per RUNBOOK.md
        _save_fs_variant("chamados_v4_fs_ftest", "airlines.txt", "1amv1ftest", "f_test", rmse=8.0)

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={"ftest": tmp_path / "chamados_v4_fs_ftest"},
        )

        row = df[df["Serie"] == "airlines"].iloc[0]
        assert row["ftest_RMSE"] == pytest.approx(8.0)  # a variante FS, nao o ARIMA copiado (12.0)


class TestBuildComparisonGeneralizedForOtherFamilies:
    """Tarefa 6-gate: BASELINE_MODEL_NAME='1amv1'/LINEAR_MODEL_NAME='1arima' eram
    constantes hardcoded, especificas da familia ARIMA-MLP hibrida -- MLP
    single usa baseline '1mlp' e NAO tem nenhum '1arima' copiado dentro de
    cada pasta chamados_v4_fs_mlp_* (SKlearnModel nao depende de modelo
    linear pre-treinado, Tarefa 5). build_comparison() precisa aceitar o
    nome do modelo baseline como parametro para nao quebrar/exigir uma copia
    da funcao a cada familia nova (SVR single, ARIMA-SVR, ...)."""

    def _save_mlp_baseline(self, base_name, rmse, tmp_path):
        fold, title = generics.format_names("chamados", base_name, "1mlp")
        generics.save_result(fold, title, [{"experiment": _FakeBaselineExperiment(rmse), "val_metric": rmse}])

    def test_baseline_model_name_parameter_selects_1mlp_instead_of_1amv1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        self._save_mlp_baseline("airlines.txt", rmse=9.0, tmp_path=tmp_path)
        _save_fs_variant("chamados_v4_fs_mlp_ftest", "airlines.txt", "1mlpftest", "f_test", rmse=7.0)

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={"ftest": tmp_path / "chamados_v4_fs_mlp_ftest"},
            baseline_model_name="1mlp",
        )

        row = df[df["Serie"] == "airlines"].iloc[0]
        assert row["Baseline_RMSE"] == pytest.approx(9.0)
        assert row["ftest_RMSE"] == pytest.approx(7.0)

    def test_default_baseline_model_name_still_1amv1_backward_compatible(self, tmp_path, monkeypatch):
        """A familia hibrida (ja em producao, results/chamados_v4_fs_comparison.csv
        real) nao pode quebrar -- o default precisa continuar '1amv1' sem
        que nenhum notebook/script existente precise passar o parametro."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        _save_baseline("sunspot.txt", rmse=5.0)
        _save_fs_variant("chamados_v4_fs_lasso", "sunspot.txt", "1amv1lasso", "lasso", rmse=4.0)

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={"lasso": tmp_path / "chamados_v4_fs_lasso"},
        )

        row = df[df["Serie"] == "sunspot"].iloc[0]
        assert row["Baseline_RMSE"] == pytest.approx(5.0)

    def test_linear_model_name_to_exclude_none_skips_the_arima_filter(self, tmp_path, monkeypatch):
        """MLP single nao copia nenhum '1arima' para dentro de
        chamados_v4_fs_mlp_* -- passar None desliga o filtro por completo,
        em vez de silenciosamente continuar procurando por '1arima' (que
        nunca estara la, mas nao deveria ser um comportamento magico)."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        self._save_mlp_baseline("austres.txt", rmse=6.0, tmp_path=tmp_path)
        _save_fs_variant("chamados_v4_fs_mlp_rfecv", "austres.txt", "1mlprfecv", "rfecv", rmse=5.5)

        df = build_comparison(
            baseline_dir=tmp_path / "chamados",
            fs_dirs={"rfecv": tmp_path / "chamados_v4_fs_mlp_rfecv"},
            baseline_model_name="1mlp",
            linear_model_name_to_exclude=None,
        )

        row = df[df["Serie"] == "austres"].iloc[0]
        assert row["rfecv_RMSE"] == pytest.approx(5.5)
