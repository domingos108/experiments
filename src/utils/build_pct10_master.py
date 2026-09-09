"""
build_pct10_master.py
----------------------
Consolida os 20 comparison.csv da matriz de Feature Selection na janela de
10% (pct10) numa tabela mestre unica -- mesmo formato/rastreabilidade de
results/benchmark_master_with_sources_v1.csv (matriz 'auto').

Nao duplica logica: reusa build_benchmark_master.build_master()/build_family_rows()
(que por sua vez reusa compare_fs_vs_baseline.build_comparison()) tal como sao,
so trocando o `baseline_dir`/`result_root` para a rodada pct10 e o mapeamento
`FAMILIES` para os nomes de modelo/diretorio dessa rodada (sufixo "pct10" nos
`.pkl`, prefixo "chamados_pct10..." nos diretorios de FS -- convencao real dos
20 notebooks `*_pct10_<estrategia>.ipynb`, RUNBOOK.md).

ARIMA fica FORA desta tabela, de proposito: o modelo ARIMA em si nao depende
de `lag_size` (nao ha janelamento de lags a selecionar -- e um modelo linear
via `auto_arima`), entao `data/result/chamados_pct10/*_1arima.pkl` e so uma
COPIA do baseline 'auto' (exigida por `Additive.fit_predict` para os hibridos
ARIMA-MLP/ARIMA-SVR, RUNBOOK.md Secao 3). Incluir uma linha ARIMA aqui
duplicaria a linha ARIMA/sem_FS ja presente na tabela 'auto', sem nenhum dado
novo -- window pct10 so existe/varia para as 4 familias que de fato fazem
janelamento de lags (MLP, SVR, ARIMA-MLP, ARIMA-SVR).

Uso
---
    python src/utils/build_pct10_master.py \
        --output results/pct10_master_v1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.build_benchmark_master import COLUMN_ORDER, FS_DEV_SERIES, build_master
from utils.export_metrics_to_csv import save_csv

DEFAULT_BASELINE_DIR = ROOT / "data" / "result" / "chamados_pct10"
DEFAULT_RESULT_ROOT = ROOT / "data" / "result"
DEFAULT_OUTPUT = ROOT / "results" / "pct10_master_v1.csv"

# As 4 familias que passaram pela janela pct10 (RUNBOOK.md: 4 baseline
# notebooks + 20 notebooks de FS = 4 arquiteturas x 5 estrategias). Nomes de
# modelo (`baseline_model_name`) e prefixo de diretorio (`fs_dir_prefix`)
# seguem a mesma convencao "sufixo pct10 sem underscore" usada pelos
# notebooks reais (ex. `1amv1pct10`, `chamados_pct10_fs_ftest`).
PCT10_FAMILIES: list[dict] = [
    {
        "name": "MLP",
        "baseline_model_name": "1mlppct10",
        "linear_model_name_to_exclude": None,
        "fs_dir_prefix": "chamados_pct10_fs_mlp",
    },
    {
        "name": "SVR",
        "baseline_model_name": "1svrpct10",
        "linear_model_name_to_exclude": None,
        "fs_dir_prefix": "chamados_pct10_fs_svr",
    },
    {
        "name": "ARIMA-MLP",
        "baseline_model_name": "1amv1pct10",
        "linear_model_name_to_exclude": "1arima",
        # Sem sufixo "arimamlp" -- mesma convencao da matriz 'auto', onde a
        # familia ARIMA-MLP usa o prefixo "nu" (chamados_v4_fs_*) e so as
        # familias mais novas (MLP/SVR/ARIMA-SVR single) ganharam um segmento
        # extra no nome (PLANO_ARQUITETURA.md Secao 5).
        "fs_dir_prefix": "chamados_pct10_fs",
    },
    {
        "name": "ARIMA-SVR",
        "baseline_model_name": "1aspct10",
        "linear_model_name_to_exclude": "1arima",
        "fs_dir_prefix": "chamados_pct10_fs_arimasvr",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gera a tabela mestre consolidada da matriz pct10 (4 familias x 6 series x "
            "6 metodos [sem_FS + 5 estrategias de FS]), reusando build_benchmark_master.build_master()."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help=f"Diretório dos 4 baselines pct10 (padrão: {DEFAULT_BASELINE_DIR}).",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=f"Diretório raiz onde vivem as pastas chamados_pct10_fs_* (padrão: {DEFAULT_RESULT_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Caminho do CSV de saída (padrão: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--series",
        type=str,
        default=",".join(FS_DEV_SERIES),
        help=(
            f"Lista de séries a incluir, separadas por vírgula (padrão: {','.join(FS_DEV_SERIES)!r}). "
            "Passe 'all' para desligar o filtro."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    series_filter = (
        None if args.series.strip().lower() == "all"
        else [s.strip() for s in args.series.split(",") if s.strip()]
    )

    df = build_master(
        args.baseline_dir.resolve(),
        args.result_root.resolve(),
        families=PCT10_FAMILIES,
        series_filter=series_filter,
    )

    assert list(df.columns) == COLUMN_ORDER  # mesmo contrato de coluna da matriz 'auto'

    save_csv(df, args.output.resolve(), label="CSV mestre consolidado (matriz pct10)")
    print("\n--- Prévia ---")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
