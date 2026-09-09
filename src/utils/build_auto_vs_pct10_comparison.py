"""
build_auto_vs_pct10_comparison.py
-----------------------------------
Cruza a matriz de Feature Selection na janela 'auto' (PACF-resolvida,
results/benchmark_master_with_sources_v1.csv) com a janela 'pct10'
(results/pct10_master_v1.csv), por Familia x Serie x Metodo_FS.

Nao recalcula NENHUMA metrica/PctGain do zero -- e um merge lado a lado dos
dois CSVs ja consolidados (cada um por sua vez ja reusa
compare_fs_vs_baseline.build_comparison(), a fonte de verdade do calculo).
`PctGain_auto`/`PctGain_pct10` sao sempre lidos direto da tabela de origem
correspondente, entao cada um ja esta calculado contra o baseline DA MESMA
janela -- nunca ha um PctGain cruzado (ex. RMSE pct10 dividido por baseline
auto) neste script.

A matriz 'auto' tem uma 5a familia (ARIMA) que nao existe em pct10_master_v1.csv
(build_pct10_master.py exclui ARIMA de proposito -- o modelo nao depende de
lag_size). O merge e `inner` por design: sem uma contraparte pct10, uma linha
ARIMA/auto nao tem o que comparar, e nao faz sentido aparecer com metade das
colunas em branco numa tabela cujo proposito e comparacao lado a lado.

Uso
---
    python src/utils/build_auto_vs_pct10_comparison.py \
        --auto-master   results/benchmark_master_with_sources_v1.csv \
        --pct10-master  results/pct10_master_v1.csv \
        --output        results/comparacao_auto_vs_pct10_v1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.export_metrics_to_csv import save_csv

DEFAULT_AUTO_MASTER = ROOT / "results" / "benchmark_master_with_sources_v1.csv"
DEFAULT_PCT10_MASTER = ROOT / "results" / "pct10_master_v1.csv"
DEFAULT_OUTPUT = ROOT / "results" / "comparacao_auto_vs_pct10_v1.csv"

KEY_COLS = ["Familia", "Serie", "Metodo_FS"]

# Colunas de cada lado que entram na comparacao lado a lado (alem das chaves).
# RMSE_FS/RMSE_Baseline (nao RMSE_mean generico) -- mesma convencao de coluna
# ja usada pelos 2 masters (build_benchmark_master.COLUMN_ORDER).
_SIDE_COLS = ["RMSE_Baseline", "RMSE_FS", "PctGain", "NFeatures", "Trivial"]

COLUMN_ORDER: list[str] = KEY_COLS + [
    "RMSE_Baseline_auto", "RMSE_Baseline_pct10",
    "RMSE_auto", "RMSE_pct10", "RMSE_Diff_pct10_menos_auto", "Janela_Vencedora",
    "PctGain_auto", "PctGain_pct10",
    "NFeatures_auto", "NFeatures_pct10",
    "Trivial_auto", "Trivial_pct10",
]


def _winner(diff: float) -> str:
    """RMSE_Diff = RMSE_pct10 - RMSE_auto: negativo = pct10 tem RMSE menor
    (pct10 venceu); positivo = auto venceu; ~0 = empate (mesmo NFeatures,
    caso trivial de 1 feature so -- ver austres/auto)."""
    if pd.isna(diff):
        return "N/A"
    if abs(diff) < 1e-9:
        return "empate"
    return "pct10" if diff < 0 else "auto"


def build_comparison(auto_master: pd.DataFrame, pct10_master: pd.DataFrame) -> pd.DataFrame:
    """Merge INNER por (Familia, Serie, Metodo_FS) -- so entra combinacao que
    exista nas DUAS janelas. `auto_master` tipicamente tem a familia ARIMA a
    mais (sem contraparte pct10, ver docstring do modulo); ela e descartada
    aqui pelo `inner`, nunca vira uma linha com metade das colunas em NaN."""
    auto = auto_master[KEY_COLS + _SIDE_COLS].rename(columns={
        "RMSE_Baseline": "RMSE_Baseline_auto",
        "RMSE_FS": "RMSE_auto",
        "PctGain": "PctGain_auto",
        "NFeatures": "NFeatures_auto",
        "Trivial": "Trivial_auto",
    })
    pct10 = pct10_master[KEY_COLS + _SIDE_COLS].rename(columns={
        "RMSE_Baseline": "RMSE_Baseline_pct10",
        "RMSE_FS": "RMSE_pct10",
        "PctGain": "PctGain_pct10",
        "NFeatures": "NFeatures_pct10",
        "Trivial": "Trivial_pct10",
    })

    merged = auto.merge(pct10, on=KEY_COLS, how="inner")

    merged["RMSE_Diff_pct10_menos_auto"] = merged["RMSE_pct10"] - merged["RMSE_auto"]
    merged["Janela_Vencedora"] = merged["RMSE_Diff_pct10_menos_auto"].apply(_winner)

    return merged[COLUMN_ORDER]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cruza a matriz 'auto' com a matriz 'pct10' por Familia x Serie x Metodo_FS, "
            "RMSE/PctGain/NFeatures lado a lado -- cada PctGain calculado contra o baseline "
            "da MESMA janela (nunca recalculado neste script)."
        )
    )
    parser.add_argument("--auto-master", type=Path, default=DEFAULT_AUTO_MASTER,
                         help=f"CSV mestre da janela 'auto' (padrão: {DEFAULT_AUTO_MASTER}).")
    parser.add_argument("--pct10-master", type=Path, default=DEFAULT_PCT10_MASTER,
                         help=f"CSV mestre da janela 'pct10' (padrão: {DEFAULT_PCT10_MASTER}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                         help=f"Caminho do CSV de saída (padrão: {DEFAULT_OUTPUT}).")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    auto_master = pd.read_csv(args.auto_master.resolve())
    pct10_master = pd.read_csv(args.pct10_master.resolve())

    df = build_comparison(auto_master, pct10_master)

    save_csv(df, args.output.resolve(), label="CSV de comparação (janela 'auto' × janela 'pct10')")
    print("\n--- Prévia ---")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
