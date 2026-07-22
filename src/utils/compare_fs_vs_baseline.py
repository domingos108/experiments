"""
compare_fs_vs_baseline.py
--------------------------
Compara o baseline ARIMA-MLP (data/result/chamados/, modelo '1amv1') contra
cada variante de Feature Selection lado a lado, por serie: RMSE de cada
variante, ganho percentual sobre o baseline, e (quando disponivel) o numero
medio de features selecionadas.

AMBIGUIDADE HISTORICA -- LEIA ANTES DE APONTAR --baseline-dir
--------------------------------------------------------------
`data/result/chamados_v2_fs_ftest/` (Tarefa 2) usa a nomenclatura ANTIGA:
o modelo la chamado `1amv1` e a PRIMEIRA execucao real do f_test, nao o
baseline sem seletor. O baseline verdadeiro para qualquer comparacao desta
Tarefa 3.1 em diante e SEMPRE `data/result/chamados/` (os 5 baselines da
Secao 3 do CLAUDE.md) -- nunca aponte --baseline-dir para uma pasta
`chamados_v*_fs_*`.

Design (Tarefa 3.1, Parte D): reaproveita export_metrics_to_csv (RMSE) e
export_selected_features (nº de features) sobre os MESMOS .pkl de cada
variante -- nao e uma terceira fonte de extracao, so um merge lado a lado
dos dois CSVs ja existentes, por serie.

Uso
---
    python src/utils/compare_fs_vs_baseline.py \
        --baseline-dir data/result/chamados \
        --fs chamados_v4_fs_ftest:data/result/chamados_v4_fs_ftest \
        --fs chamados_v4_fs_mutualinfo:data/result/chamados_v4_fs_mutualinfo \
        --fs chamados_v4_fs_rfembedded:data/result/chamados_v4_fs_rfembedded \
        --fs chamados_v4_fs_lasso:data/result/chamados_v4_fs_lasso \
        --output results/chamados_v4_fs_comparison.csv
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

from utils.export_metrics_to_csv import aggregate_mean as _aggregate_metrics_mean
from utils.export_metrics_to_csv import collect_all_rows as _collect_metrics_rows
from utils.export_metrics_to_csv import save_csv
from utils.export_selected_features import aggregate_mean as _aggregate_features_mean
from utils.export_selected_features import collect_all_rows as _collect_features_rows

DEFAULT_OUTPUT = ROOT / "results" / "fs_vs_baseline_comparison.csv"
BASELINE_MODEL_NAME = "1amv1"

# RUNBOOK.md Secao 3/8b manda copiar <serie>_1arima.pkl para DENTRO de cada
# pasta chamados_v4_fs_* (Additive precisa do ARIMA pre-treinado sob o mesmo
# experiment_id). Todo result_dir de FS real tem, portanto, 2 modelos por
# serie -- o ARIMA copiado E a variante FS -- e precisa ser filtrado antes de
# indexar por Serie, senao vira indice duplicado.
LINEAR_MODEL_NAME = "1arima"


def _baseline_rmse_by_serie(baseline_dir: Path, metric: str, baseline_model_name: str) -> pd.Series:
    detail = _collect_metrics_rows(baseline_dir)
    if detail.empty:
        raise ValueError(f"Nenhum resultado encontrado em {baseline_dir}")

    agg = _aggregate_metrics_mean(detail)
    agg = agg[agg["Modelo"] == baseline_model_name]
    if agg.empty:
        raise ValueError(
            f"Nenhum resultado do modelo '{baseline_model_name}' encontrado em {baseline_dir}. "
            "Confirme que --baseline-dir aponta para data/result/chamados (nunca uma pasta "
            "chamados_v*_fs_*, ver ambiguidade historica no topo deste arquivo) e que "
            "baseline_model_name corresponde a familia sendo comparada (ex. '1amv1' para "
            "ARIMA-MLP hibrido, '1mlp' para MLP single)."
        )
    return agg.set_index("Serie")[metric]


def _fs_rmse_by_serie(result_dir: Path, metric: str, linear_model_name_to_exclude: str | None) -> pd.Series:
    detail = _collect_metrics_rows(result_dir)
    if detail.empty:
        return pd.Series(dtype=float)
    agg = _aggregate_metrics_mean(detail)
    if linear_model_name_to_exclude is not None:
        agg = agg[agg["Modelo"] != linear_model_name_to_exclude]
    return agg.set_index("Serie")[metric]


def _fs_n_features_by_serie(result_dir: Path) -> pd.Series:
    detail = _collect_features_rows(result_dir)
    if detail.empty:
        return pd.Series(dtype=float)
    agg = _aggregate_features_mean(detail)
    return agg.set_index("Serie")["N_Features_Selected_mean"]


def build_comparison(
    baseline_dir: Path,
    fs_dirs: dict[str, Path],
    metric: str = "RMSE_mean",
    baseline_model_name: str = BASELINE_MODEL_NAME,
    linear_model_name_to_exclude: str | None = LINEAR_MODEL_NAME,
) -> pd.DataFrame:
    """
    Retorna um DataFrame com uma linha por serie do baseline e, para cada
    `strategy` em `fs_dirs`, as colunas `<strategy>_RMSE`,
    `<strategy>_PctGain` (positivo = variante melhor que o baseline) e,
    quando disponivel, `<strategy>_NFeatures` (media de features
    selecionadas nas repeticoes daquela serie).

    `baseline_model_name`/`linear_model_name_to_exclude` sao parametrizaveis
    (defaults preservam o comportamento original da familia ARIMA-MLP
    hibrida: '1amv1'/'1arima') para reuso por outras familias sem duplicar
    esta funcao -- ex. MLP single usa baseline_model_name='1mlp' e
    linear_model_name_to_exclude=None (SKlearnModel nao copia nenhum modelo
    linear pre-treinado para dentro do result_dir de FS, Tarefa 5).
    """
    baseline_rmse = _baseline_rmse_by_serie(baseline_dir, metric, baseline_model_name)

    df = pd.DataFrame({"Serie": baseline_rmse.index, "Baseline_RMSE": baseline_rmse.values})

    for strategy, result_dir in fs_dirs.items():
        fs_rmse = _fs_rmse_by_serie(result_dir, metric, linear_model_name_to_exclude)
        df[f"{strategy}_RMSE"] = df["Serie"].map(fs_rmse)
        df[f"{strategy}_PctGain"] = (
            (df["Baseline_RMSE"] - df[f"{strategy}_RMSE"]) / df["Baseline_RMSE"] * 100
        )

        n_features = _fs_n_features_by_serie(result_dir)
        if not n_features.empty:
            df[f"{strategy}_NFeatures"] = df["Serie"].map(n_features)

    return df


def _parse_fs_arg(raw: str) -> tuple[str, Path]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"--fs precisa ser 'strategy_label:caminho', recebido: {raw!r}"
        )
    label, path_str = raw.split(":", maxsplit=1)
    return label, Path(path_str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara o baseline ARIMA-MLP (data/result/chamados/, modelo '1amv1') "
            "contra cada variante de Feature Selection, lado a lado por serie."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Diretório do baseline (sempre data/result/chamados -- ver ambiguidade histórica no topo deste arquivo).",
    )
    parser.add_argument(
        "--fs",
        action="append",
        type=_parse_fs_arg,
        required=True,
        dest="fs_dirs",
        metavar="LABEL:PATH",
        help="Variante de FS a comparar, no formato 'rotulo:diretorio'. Repetível.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Caminho do CSV de saída (padrão: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--baseline-model-name",
        type=str,
        default=BASELINE_MODEL_NAME,
        help=f"Sufixo do modelo baseline dentro de --baseline-dir (padrão: {BASELINE_MODEL_NAME!r} -- ARIMA-MLP híbrido; use '1mlp' para MLP single, etc.).",
    )
    parser.add_argument(
        "--linear-model-name-to-exclude",
        type=str,
        default=LINEAR_MODEL_NAME,
        help=(
            f"Modelo linear copiado para dentro de cada pasta de FS a ignorar na comparação "
            f"(padrão: {LINEAR_MODEL_NAME!r}). Passe string vazia para desligar o filtro "
            "(famílias single-model, como MLP, não copiam nenhum modelo linear)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    fs_dirs = dict(args.fs_dirs)
    linear_model_name_to_exclude = args.linear_model_name_to_exclude or None
    df = build_comparison(
        args.baseline_dir.resolve(),
        {k: v.resolve() for k, v in fs_dirs.items()},
        baseline_model_name=args.baseline_model_name,
        linear_model_name_to_exclude=linear_model_name_to_exclude,
    )

    save_csv(df, args.output.resolve(), label="CSV de comparação (baseline × variantes FS)")
    print("\n--- Prévia ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
