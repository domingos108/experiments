"""
Testes de src/utils/build_tabela_orientador.py (Tarefa 14).

Cobre so a logica pura de selecao/ordenacao de linhas e de negrito do menor
RMSE por serie -- nao le results/benchmark_master_with_sources_v1.csv real
(isso e responsabilidade de build_tabela_orientador(), testada manualmente
via CLI, mesmo padrao de compare_fs_vs_baseline.py).
"""

import pandas as pd

from utils.build_tabela_orientador import mark_bold, select_table_rows


def _row(familia, serie, metodo, rmse=1.0, mape=1.0, nfeatures=5.0, trivial=False, fonte="src.csv"):
    return {
        "Familia": familia,
        "Serie": serie,
        "Metodo_FS": metodo,
        "RMSE": rmse,
        "MAPE": mape,
        "NFeatures": nfeatures,
        "Trivial": trivial,
        "Fonte": fonte,
    }


class TestSelectTableRows:
    def test_reference_families_keep_only_sem_fs_row(self):
        # SVR tem linhas de FS na tabela mestre (Tarefa 9 rodou FS para SVR
        # single tambem), mas o escopo desta tabela e so referencia sem FS
        # para familias single -- a linha 'ftest' precisa ser descartada.
        df = pd.DataFrame([
            _row("SVR", "airlines", "sem_FS", rmse=10.0),
            _row("SVR", "airlines", "ftest", rmse=9.0),
            _row("ARIMA-MLP", "airlines", "sem_FS", rmse=8.0),
        ])

        result = select_table_rows(df)

        svr_rows = result[result["Familia"] == "SVR"]
        assert list(svr_rows["Metodo_FS"]) == ["sem_FS"]

    def test_hybrid_family_methods_ordered_by_method_order_not_csv_order(self):
        # Embaralhado de proposito -- a ordem de saida nao pode depender da
        # ordem incidental de linhas no CSV de origem.
        df = pd.DataFrame([
            _row("ARIMA-MLP", "airlines", "rfecv"),
            _row("ARIMA-MLP", "airlines", "lasso"),
            _row("ARIMA-MLP", "airlines", "ftest"),
            _row("ARIMA-MLP", "airlines", "sem_FS"),
            _row("ARIMA-MLP", "airlines", "rfembedded"),
            _row("ARIMA-MLP", "airlines", "mutualinfo"),
        ])

        result = select_table_rows(df)

        assert list(result["Metodo_FS"]) == [
            "sem_FS", "ftest", "mutualinfo", "rfembedded", "lasso", "rfecv",
        ]


class TestMarkBold:
    def test_flags_every_row_tied_at_series_minimum_scoped_per_serie(self):
        df = pd.DataFrame([
            _row("ARIMA-SVR", "airlines", "sem_FS", rmse=15.1665),
            _row("ARIMA-SVR", "airlines", "ftest", rmse=15.1665),
            _row("ARIMA-SVR", "airlines", "rfecv", rmse=15.4463),
            _row("MLP", "airlines", "sem_FS", rmse=27.6977),
        ])

        result = mark_bold(df)

        assert result.loc[result["Familia"] == "ARIMA-SVR", "Bold"].tolist() == [True, True, False]
        assert result.loc[result["Familia"] == "MLP", "Bold"].tolist() == [False]
