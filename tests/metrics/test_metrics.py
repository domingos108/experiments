"""
Teste de regressao do bug de separador de caminho em metrics.py.

open_fold_result() extrai o nome da serie a partir do caminho retornado por
glob(). No Linux/Mac, glob() sempre usa '/'. No Windows, glob() insere '\\'
entre o diretorio e o nome do arquivo mesmo quando o diretorio-base foi
construido com '/' (caso de config.MODEL_DATA_PATH, que mistura os dois
estilos) -- pth.split('/')[-1] entao falha silenciosamente, extraindo o
nome da PASTA (ou parte dela) em vez do nome real da serie.

Isso quebrava metrics.open_fold_result()/calculate_metrics_v2.ipynb no
Windows para QUALQUER experiment_id, incluindo os 5 baselines originais
(ver relatorio da Tarefa 2, item 5).
"""

import numpy as np

import config
from metrics.metrics import extract_series_name_from_path, open_fold_result
from model import generics


def _build_fake_pkl_result(rmse=1.0, n_test=3):
    """Constroi um resultado no mesmo formato que GridSearch.execution() salva
    de verdade, sem depender de treinar nenhum modelo real."""
    metrics_results = {
        "test_metrics": {
            "RMSE": rmse, "MSE": rmse ** 2, "MAE": rmse, "MAPE": rmse,
            "theil": 1.0, "ARV": 1.0, "IA": 1.0, "POCID": 50.0,
        },
        "time_exec": {"training": 0.1, "testing": 0.01},
        "test_predict": np.zeros(n_test),
    }
    return [{"experiment": generics.ResultExp(metrics_results), "val_metric": rmse}]


class TestPklNamingConventionCompatibility:
    """Prova de integracao (Tarefa 3, Parte B): o sufixo de estrategia sem
    underscore (ex: '1amv1ftest') tem que ser lido corretamente por
    open_fold_result() SEM nenhuma mudanca em metrics.py. Nunca toca em
    data/result/ real -- usa um experiment_id temporario isolado."""

    def test_suffixed_model_name_is_extracted_without_truncation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        fold, title = generics.format_names("fake_fs_experiment", "sunspot.txt", "1amv1ftest")
        generics.save_result(fold, title, _build_fake_pkl_result(rmse=2.5))

        df_mean, df_all, df_prevs = open_fold_result("fake_fs_experiment")

        assert ("sunspot", "1amv1ftest") in df_mean.index
        assert df_mean.loc[("sunspot", "1amv1ftest"), "RMSE"] == 2.5

    def test_four_different_strategy_suffixes_do_not_collide_or_truncate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        for suffix, rmse in [("ftest", 1.0), ("mutualinfo", 2.0), ("rfembedded", 3.0), ("lasso", 4.0)]:
            fold, title = generics.format_names("fake_fs_experiment", "airlines.txt", f"1amv1{suffix}")
            generics.save_result(fold, title, _build_fake_pkl_result(rmse=rmse))

        df_mean, df_all, df_prevs = open_fold_result("fake_fs_experiment")

        for suffix, rmse in [("ftest", 1.0), ("mutualinfo", 2.0), ("rfembedded", 3.0), ("lasso", 4.0)]:
            assert ("airlines", f"1amv1{suffix}") in df_mean.index
            assert df_mean.loc[("airlines", f"1amv1{suffix}"), "RMSE"] == rmse


class TestFiveBaselinesNamingRegression:
    """O sufixo novo e exclusivo de notebooks com Pipeline/seletor -- os 5
    baselines da Secao 3 do CLAUDE.md continuam sem sufixo, nome e
    comportamento identicos. Simula (sem depender de .pkl real em disco) o
    padrao de nome exato que os 5 baselines usam hoje."""

    def test_baseline_style_names_still_resolve_to_bare_model_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        for base_name, model_name in [
            ("airlines.txt", "1amv1"), ("airlines.txt", "1arima"),
            ("airlines.txt", "1as"), ("airlines.txt", "1mlp"), ("airlines.txt", "1svr"),
        ]:
            fold, title = generics.format_names("fake_baseline_experiment", base_name, model_name)
            generics.save_result(fold, title, _build_fake_pkl_result(rmse=1.0))

        df_mean, df_all, df_prevs = open_fold_result("fake_baseline_experiment")

        for model_name in ["1amv1", "1arima", "1as", "1mlp", "1svr"]:
            assert ("airlines", model_name) in df_mean.index


class TestExtractSeriesNameFromPath:
    def test_unix_style_path(self):
        pth = "/home/user/experiments/data/result/chamados/airlines_1amv1.pkl"
        assert extract_series_name_from_path(pth) == "airlines"

    def test_windows_style_path_pure_backslash(self):
        pth = r"C:\Projetos\experiments\data\result\chamados\airlines_1amv1.pkl"
        assert extract_series_name_from_path(pth) == "airlines"

    def test_windows_style_mixed_separators_reproducing_the_real_bug(self):
        """Reproduz exatamente o caminho que glob() produz neste projeto no
        Windows: base construida com '/' (config.MODEL_DATA_PATH), glob()
        inserindo '\\' antes do nome do arquivo. A logica antiga
        (pth.split('/')[-1]) extraia 'chamados' em vez de 'airlines' --
        exatamente o bug relatado na Tarefa 2."""
        pth = (
            r"C:\Projetos\mestrado_codigos\experiments"
            "/data/result/chamados_v2_fs_ftest"
            r"\airlines_1amv1.pkl"
        )
        assert extract_series_name_from_path(pth) == "airlines"

    def test_series_name_with_underscore_in_experiment_id_does_not_leak_in(self):
        """experiment_id com underscore (ex: chamados_v2_fs_ftest) nao pode
        contaminar o nome da serie extraido -- essa era a manifestacao
        exata do bug."""
        pth = r"C:\...\data\result\chamados_v2_fs_ftest\sunspot_1amv1.pkl"
        assert extract_series_name_from_path(pth) == "sunspot"
