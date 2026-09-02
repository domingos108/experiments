"""
build_benchmark_master.py
--------------------------
Formaliza como script versionado a geracao da tabela mestre consolidada
(historicamente results/benchmark_master_with_sources_v1.csv, gerada por um
script Python executado fora do Claude Code, nunca versionado -- ver
CLAUDE.md). Reusa compare_fs_vs_baseline.build_comparison() por familia (uma
chamada por metrica em METRIC_KEYS, ja que build_comparison() so expoe uma
metrica -- e sua respectiva coluna de PctGain -- por vez) e consolida a
saida em formato LONG: uma linha por Familia x Serie x Metodo_FS, com as
colunas de rastreamento de origem (Metrics_Source_File, Pkl_Source_Dir,
Features_Source_File) presentes no CSV historico.

`PctGain` e `NFeatures` sao sempre derivados da chamada com metric='RMSE_mean'
(mesma convencao do CSV historico: ganho percentual e sempre sobre RMSE,
mesmo quando a linha tambem reporta MSE/MAE/MAPE/theil/ARV/IA/POCID).

Uso
---
    python src/utils/build_benchmark_master.py \
        --baseline-dir data/result/chamados \
        --result-root  data/result \
        --output       results/benchmark_master.csv
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

from utils.compare_fs_vs_baseline import build_comparison
from utils.export_metrics_to_csv import METRIC_KEYS, save_csv
from utils.export_selected_features import aggregate_mean as _aggregate_features_mean
from utils.export_selected_features import collect_all_rows as _collect_features_rows

DEFAULT_BASELINE_DIR = ROOT / "data" / "result" / "chamados"
DEFAULT_RESULT_ROOT = ROOT / "data" / "result"
DEFAULT_OUTPUT = ROOT / "results" / "benchmark_master.csv"

# 5 estrategias do arsenal (PLANO_ARQUITETURA.md Secao 2) -- rotulo de
# diretorio sem underscore, mesma convencao ja usada pelos notebooks reais
# (Tarefa 5, Secao 1.8) e por compare_fs_vs_baseline.py.
STRATEGIES = ["ftest", "mutualinfo", "rfembedded", "lasso", "rfecv"]

# Classificacao documentada (RUNBOOK.md Secao 1, results/tabela_consolidada_
# orientador.md: "austres -- serie trivial, 1 lag candidato"): austres.txt
# resolve lag_size='auto'=1, sem espaco real de selecao de features.
# Constante explicita por serie (nao derivada de NFeatures==1) porque a
# familia ARIMA nao tem NFeatures (nao ha janelamento/FS nela) e ainda assim
# precisa do mesmo flag.
TRIVIAL_SERIES = {"austres"}

# Escopo atual da fase de FS (RUNBOOK.md Secao 1: "Hoje restrito a
# FS_DEV_SERIES... Expandir a lista alem dessas 4 e decisao explicita
# futura, nao algo a fazer implicitamente"). data/result/chamados/ guarda
# baselines de 17 series (Secao 3.4 abaixo), mas so estas 4 tem variantes de
# FS -- sem este filtro, a linha sem_FS vazaria para as outras 13 series,
# expandindo o escopo da tabela em silencio. Duplicado (nao importado) de
# tests/model/conftest.py -- mesmo precedente ja usado em
# src/utils/audit_experiment_integrity.py.
FS_DEV_SERIES = ["airlines", "austres", "coloradoRiver", "sunspot",
                 "windspeedfortaleza", "samurec"]

# Uma familia por linha da Secao 3 do CLAUDE.md (5 baselines protegidos) +
# suas variantes de FS (Tarefas 4-8/PLANO_ARQUITETURA.md). `fs_dir_prefix`
# None = familia sem Feature Selection (ARIMA) -- so a linha sem_FS existe.
FAMILIES: list[dict] = [
    {
        "name": "ARIMA",
        "baseline_model_name": "1arima",
        "linear_model_name_to_exclude": None,
        "fs_dir_prefix": None,
    },
    {
        "name": "MLP",
        "baseline_model_name": "1mlp",
        "linear_model_name_to_exclude": None,
        "fs_dir_prefix": "chamados_v4_fs_mlp",
    },
    {
        "name": "SVR",
        "baseline_model_name": "1svr",
        "linear_model_name_to_exclude": None,
        "fs_dir_prefix": "chamados_v4_fs_svr",
    },
    {
        "name": "ARIMA-MLP",
        "baseline_model_name": "1amv1",
        "linear_model_name_to_exclude": "1arima",
        "fs_dir_prefix": "chamados_v4_fs",
    },
    {
        "name": "ARIMA-SVR",
        "baseline_model_name": "1as",
        "linear_model_name_to_exclude": "1arima",
        "fs_dir_prefix": "chamados_v4_fs_arimasvr",
    },
]

COLUMN_ORDER: list[str] = (
    ["Familia", "Serie", "Metodo_FS", "Metrics_Source_File", "Pkl_Source_Dir",
     "Features_Source_File", "NFeatures", "Trivial"]
    + [f"{key}_{suffix}" for key in METRIC_KEYS for suffix in ("Baseline", "FS")]
    + ["PctGain"]
)


def fs_dirs_for_family(result_root: Path, fs_dir_prefix: str | None) -> dict[str, Path]:
    """Expande o prefixo de diretorio de uma familia nas 5 pastas de FS
    (uma por estrategia). Familias sem FS (ARIMA) usam prefix=None -> {}."""
    if fs_dir_prefix is None:
        return {}
    return {strategy: result_root / f"{fs_dir_prefix}_{strategy}" for strategy in STRATEGIES}


def _lookup(df: pd.DataFrame, serie: str, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    match = df.loc[df["Serie"] == serie, col]
    return float(match.iloc[0]) if not match.empty else float("nan")


def _n_features_total_by_serie(fs_dirs: dict[str, Path]) -> pd.Series:
    """N_Features_Total (lag_size='auto' resolvido) e constante por serie
    dentro de uma familia, independente da estrategia de FS -- mesmo residuo
    do ARIMA/mesma serie bruta alimenta todos os seletores (PLANO_ARQUITETURA.md
    Secao 1.8). Usa a primeira pasta de FS que tiver dados; se a primeira
    (ftest) ainda nao foi rodada, tenta as seguintes -- nao assume que TODAS
    as 5 estrategias ja existem."""
    for result_dir in fs_dirs.values():
        detail = _collect_features_rows(result_dir)
        if detail.empty:
            continue
        agg = _aggregate_features_mean(detail)
        return agg.groupby("Serie")["N_Features_Total"].first()
    return pd.Series(dtype=float)


def _baseline_row(
    family_name: str,
    serie: str,
    per_metric: dict[str, pd.DataFrame],
    n_features_total: pd.Series,
    baseline_dir: Path,
) -> dict:
    row = {
        "Familia": family_name,
        "Serie": serie,
        "Metodo_FS": "sem_FS",
        "Metrics_Source_File": "results/baseline_metrics.csv",
        "Pkl_Source_Dir": f"data/result/{baseline_dir.name}/",
        "Features_Source_File": "",
        "NFeatures": float(n_features_total.get(serie, float("nan"))) if not n_features_total.empty else float("nan"),
        "Trivial": serie in TRIVIAL_SERIES,
        "PctGain": 0.0,
    }
    for key in METRIC_KEYS:
        value = _lookup(per_metric[key], serie, "Baseline_RMSE")
        row[f"{key}_Baseline"] = value
        row[f"{key}_FS"] = value
    return row


def _fs_row(
    family_name: str,
    serie: str,
    strategy: str,
    fs_dir: Path,
    per_metric: dict[str, pd.DataFrame],
) -> dict:
    rmse_df = per_metric["RMSE"]
    row = {
        "Familia": family_name,
        "Serie": serie,
        "Metodo_FS": strategy,
        "Metrics_Source_File": f"results/{fs_dir.name}/metrics.csv",
        "Pkl_Source_Dir": f"data/result/{fs_dir.name}/",
        "Features_Source_File": f"results/{fs_dir.name}/selected_features.csv",
        "NFeatures": _lookup(rmse_df, serie, f"{strategy}_NFeatures"),
        "Trivial": serie in TRIVIAL_SERIES,
        "PctGain": _lookup(rmse_df, serie, f"{strategy}_PctGain"),
    }
    for key in METRIC_KEYS:
        df = per_metric[key]
        row[f"{key}_Baseline"] = _lookup(df, serie, "Baseline_RMSE")
        row[f"{key}_FS"] = _lookup(df, serie, f"{strategy}_RMSE")
    return row


def build_family_rows(
    baseline_dir: Path,
    result_root: Path,
    family: dict,
    series_filter: list[str] | set[str] | None = None,
) -> list[dict]:
    """Retorna as linhas (formato LONG) de uma unica familia: 1 linha sem_FS
    por serie do baseline, + 1 linha por (serie, estrategia) cuja pasta de FS
    exista e tenha dados -- estrategias sem pasta/sem .pkl ainda nao viram
    linha (nao ha o que reportar), mesmo invariante de "falta dado -> NaN
    dentro da linha, nunca linha fantasma" ja usado por build_comparison().

    `series_filter`, quando informado, restringe as series do baseline
    consideradas (ex.: FS_DEV_SERIES) -- baseline_dir tipicamente contem MAIS
    series do que as que tem variante de FS (data/result/chamados/ guarda 17
    baselines, so 4 tem pastas chamados_v4_fs_*), e sem o filtro a linha
    sem_FS vazaria escopo para series que a fase de FS ainda nao cobre."""
    fs_dirs = fs_dirs_for_family(result_root, family["fs_dir_prefix"])

    per_metric = {
        key: build_comparison(
            baseline_dir,
            fs_dirs,
            metric=f"{key}_mean",
            baseline_model_name=family["baseline_model_name"],
            linear_model_name_to_exclude=family["linear_model_name_to_exclude"],
        )
        for key in METRIC_KEYS
    }

    n_features_total = _n_features_total_by_serie(fs_dirs)

    series_list = per_metric["RMSE"]["Serie"].tolist()
    if series_filter is not None:
        series_list = [s for s in series_list if s in series_filter]

    rows: list[dict] = []
    for serie in series_list:
        rows.append(_baseline_row(family["name"], serie, per_metric, n_features_total, baseline_dir))

        for strategy, fs_dir in fs_dirs.items():
            fs_rmse_col = f"{strategy}_RMSE"
            if fs_rmse_col not in per_metric["RMSE"].columns:
                continue
            if pd.isna(_lookup(per_metric["RMSE"], serie, fs_rmse_col)):
                continue
            rows.append(_fs_row(family["name"], serie, strategy, fs_dir, per_metric))

    return rows


def build_master(
    baseline_dir: Path,
    result_root: Path,
    families: list[dict] | None = None,
    series_filter: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    families = FAMILIES if families is None else families

    rows: list[dict] = []
    for family in families:
        rows.extend(build_family_rows(baseline_dir, result_root, family, series_filter=series_filter))

    df = pd.DataFrame(rows)
    return df[COLUMN_ORDER]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gera a tabela mestre consolidada (formato LONG, todas as familias x "
            "series x metodos de FS), reusando compare_fs_vs_baseline.build_comparison() "
            "por familia."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help=f"Diretório dos 5 baselines protegidos (padrão: {DEFAULT_BASELINE_DIR}).",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=f"Diretório raiz onde vivem as pastas chamados_v4_fs_* (padrão: {DEFAULT_RESULT_ROOT}).",
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
            f"Lista de séries a incluir, separadas por vírgula (padrão: {','.join(FS_DEV_SERIES)!r} -- "
            "escopo atual da fase de FS, RUNBOOK.md Seção 1). Passe 'all' para desligar o filtro "
            "e incluir toda série presente em --baseline-dir (expande o escopo -- use com cautela)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    series_filter = None if args.series.strip().lower() == "all" else [s.strip() for s in args.series.split(",") if s.strip()]

    df = build_master(args.baseline_dir.resolve(), args.result_root.resolve(), series_filter=series_filter)

    save_csv(df, args.output.resolve(), label="CSV mestre consolidado (todas as familias)")
    print("\n--- Prévia ---")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
