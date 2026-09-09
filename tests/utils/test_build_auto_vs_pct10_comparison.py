"""
Testes de src/utils/build_auto_vs_pct10_comparison.py.

Nao recalcula RMSE/PctGain -- so testa o merge (Familia, Serie, Metodo_FS) e a
derivacao de RMSE_Diff/Janela_Vencedora a partir de colunas ja calculadas por
build_benchmark_master.py/build_pct10_master.py (mockadas aqui como DataFrames
minimos, sem tocar nenhum .pkl real).
"""

import numpy as np
import pandas as pd
import pytest

from utils.build_auto_vs_pct10_comparison import COLUMN_ORDER, build_comparison


def _row(familia, serie, metodo, rmse_baseline, rmse_fs, pctgain, nfeatures, trivial=False):
    return {
        "Familia": familia, "Serie": serie, "Metodo_FS": metodo,
        "RMSE_Baseline": rmse_baseline, "RMSE_FS": rmse_fs,
        "PctGain": pctgain, "NFeatures": nfeatures, "Trivial": trivial,
    }


class TestBuildComparison:
    def test_inner_join_drops_family_only_present_in_auto(self):
        auto = pd.DataFrame([
            _row("ARIMA", "airlines", "sem_FS", 19.0, 19.0, 0.0, np.nan),
            _row("MLP", "airlines", "ftest", 27.0, 21.0, 22.2, 5),
        ])
        pct10 = pd.DataFrame([
            _row("MLP", "airlines", "ftest", 28.8, 21.7, 24.6, 5),
        ])

        df = build_comparison(auto, pct10)

        assert list(df.columns) == COLUMN_ORDER
        assert len(df) == 1
        assert df.iloc[0]["Familia"] == "MLP"  # ARIMA descartado (sem par pct10)

    def test_rmse_diff_and_winner_when_pct10_has_lower_rmse(self):
        auto = pd.DataFrame([_row("ARIMA-MLP", "austres", "rfecv", 20.775953, 20.775953, 0.0, np.nan, trivial=True)])
        pct10 = pd.DataFrame([_row("ARIMA-MLP", "austres", "rfecv", 20.775953, 20.529176, 1.187799, 6.2, trivial=False)])

        df = build_comparison(auto, pct10)
        row = df.iloc[0]

        expected_diff = 20.529176 - 20.775953
        assert row["RMSE_Diff_pct10_menos_auto"] == pytest.approx(expected_diff)
        assert row["Janela_Vencedora"] == "pct10"
        assert row["RMSE_auto"] == pytest.approx(20.775953)
        assert row["RMSE_pct10"] == pytest.approx(20.529176)
        assert row["PctGain_auto"] == pytest.approx(0.0)
        assert row["PctGain_pct10"] == pytest.approx(1.187799)

    def test_winner_is_auto_when_auto_has_lower_rmse(self):
        auto = pd.DataFrame([_row("SVR", "sunspot", "ftest", 22.0, 19.0, 13.6, 9)])
        pct10 = pd.DataFrame([_row("SVR", "sunspot", "ftest", 22.0, 19.26, 12.4, 9)])

        df = build_comparison(auto, pct10)
        assert df.iloc[0]["Janela_Vencedora"] == "auto"

    def test_winner_is_empate_when_rmse_is_identical(self):
        auto = pd.DataFrame([_row("SVR", "austres", "lasso", 21.0, 21.0, 0.0, 1, trivial=True)])
        pct10 = pd.DataFrame([_row("SVR", "austres", "lasso", 21.078692, 21.0, 0.373, 8, trivial=False)])

        df = build_comparison(auto, pct10)
        assert df.iloc[0]["Janela_Vencedora"] == "empate"

    def test_nfeatures_and_trivial_carried_from_each_side_independently(self):
        """Caso austres: NFeatures diverge muito entre janelas (1 no 'auto',
        ~8 no pct10) e Trivial deixa de valer para pct10 -- ambas as colunas
        devem sobreviver ao merge, uma por janela, sem serem confundidas."""
        auto = pd.DataFrame([_row("MLP", "austres", "ftest", 27.7, 27.7, 0.0, 1, trivial=True)])
        pct10 = pd.DataFrame([_row("MLP", "austres", "ftest", 24.680016, 22.239701, 9.887820, 1.0, trivial=True)])

        df = build_comparison(auto, pct10)
        row = df.iloc[0]

        assert row["NFeatures_auto"] == 1
        assert row["NFeatures_pct10"] == 1.0
        assert row["Trivial_auto"] == True
        assert row["Trivial_pct10"] == True

    def test_pctgain_columns_are_never_recomputed_only_carried_over(self):
        """PctGain de cada lado deve vir exatamente da tabela de origem --
        nunca recalculado contra o baseline errado (ex. RMSE pct10 dividido
        pelo baseline auto)."""
        auto = pd.DataFrame([_row("ARIMA-SVR", "coloradoRiver", "lasso", 0.39, 0.35, 10.25641, 40)])
        pct10 = pd.DataFrame([_row("ARIMA-SVR", "coloradoRiver", "lasso", 0.41, 0.30, 26.82927, 55)])

        df = build_comparison(auto, pct10)
        row = df.iloc[0]

        assert row["PctGain_auto"] == pytest.approx(10.25641)
        assert row["PctGain_pct10"] == pytest.approx(26.82927)
