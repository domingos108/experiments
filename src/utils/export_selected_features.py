"""
export_selected_features.py
----------------------------
Varre um diretorio de resultados (data/result/<experiment_id>/), le os .pkl
gerados pelo pipeline de Feature Selection (Pipeline([('selector',
TimeSeriesFeatureSelector), ('estimator', ...)])) e compila o numero/indices
de features selecionadas por execucao em CSV.

Design (Tarefa 3.1, PLANO_ARQUITETURA.md Secao 1.5): extracao POS-HOC, sem
qualquer mudanca em generics.py -- o Pipeline fitted (com o seletor ja
ajustado) ja sobrevive dentro do .pkl hoje, porque Additive/SKlearnModel
guardam `self.model` (mutado in place por generics.fit_predict_ml_schemma) e
GridSearch.execution() serializa a instancia inteira. .pkl sem seletor (os 5
baselines da Secao 3 do CLAUDE.md, ou qualquer wrapper que nao use Pipeline)
sao ignorados com aviso, nunca erro.

Uso
---
    python src/utils/export_selected_features.py \
        --result-dir data/result/chamados_v4_fs_rfembedded \
        --output      results/chamados_v4_fs_rfembedded/selected_features.csv \
        --detail
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

import config
from utils.export_metrics_to_csv import _load_pkl, _unwrap_entry, parse_pkl_path, save_csv

DEFAULT_RESULT_DIR = Path(config.MODEL_DATA_PATH)
DEFAULT_OUTPUT = ROOT / "results" / "selected_features.csv"


def _extract_selector(result_exp):
    """Retorna o TimeSeriesFeatureSelector fitted do Pipeline, ou None se
    `result_exp` nao tiver um Pipeline com step 'selector' ja ajustado com o
    que este script precisa. Cobre 3 casos, todos como skip silencioso (nao
    erro): (1) os 5 baselines, cujo `.model` e um estimador simples; (2) um
    Pipeline sem step 'selector'; (3) um selector fitted por uma versao PRE
    Tarefa 3.1 do TimeSeriesFeatureSelector, que nao setava `n_features_in_`
    (ex: data/result/chamados_v2_fs_ftest/, ja executado de verdade na
    Tarefa 2 -- ver PLANO_ARQUITETURA.md Secao 1.5)."""
    model = getattr(result_exp, "model", None)
    if model is None or not hasattr(model, "named_steps"):
        return None
    selector = model.named_steps.get("selector")
    if selector is None:
        return None
    if not hasattr(selector, "selected_indices_") or not hasattr(selector, "n_features_in_"):
        return None
    return selector


def _lag_names(selected_indices, n_features_total: int) -> list[str]:
    """create_windowing/input.open_format_train_val_test nomeiam as colunas
    lag_L, lag_{L-1}, ..., lag_1 (L = n_features_total, horizon=1 em todos os
    notebooks de FS) -- a coluna de indice posicional j corresponde a
    lag_{L-j} (ver PLANO_ARQUITETURA.md Secao 1.5)."""
    return [f"lag_{n_features_total - idx}" for idx in selected_indices]


def extract_rows(pkl_path: Path) -> list[dict]:
    """
    Le um .pkl e retorna uma lista de dicionarios (uma linha por repeticao),
    uma por entrada cujo `.model` seja um Pipeline com um seletor ja
    ajustado. Entradas sem seletor (baselines) sao silenciosamente
    ignoradas -- nao geram linha nem erro.
    """
    experiment_id, serie, modelo = parse_pkl_path(pkl_path)

    try:
        entries = _load_pkl(pkl_path)
    except Exception as exc:
        print(f"  [AVISO] Não foi possível ler {pkl_path.name}: {exc}")
        return []

    rows: list[dict] = []

    for repetition_idx, entry in enumerate(entries):
        try:
            result_exp, _val_metric = _unwrap_entry(entry)

            selector = _extract_selector(result_exp)
            if selector is None:
                continue

            n_total = int(selector.n_features_in_)
            selected = [int(i) for i in selector.selected_indices_.tolist()]

            rows.append({
                "ExperimentID": experiment_id,
                "Serie": serie,
                "Modelo": modelo,
                "Repeticao": repetition_idx,
                "Strategy": selector.strategy,
                "N_Features_Selected": len(selected),
                "N_Features_Total": n_total,
                "Selected_Indices": ";".join(str(i) for i in selected),
                "Selected_Lag_Names": ";".join(_lag_names(selected, n_total)),
            })

        except Exception as exc:
            print(
                f"  [AVISO] Erro na repetição {repetition_idx} "
                f"de {pkl_path.name}: {exc}"
            )

    return rows


def collect_all_rows(result_dir: Path) -> pd.DataFrame:
    """Varre `result_dir` recursivamente e retorna um DataFrame com uma linha
    por repeticao por .pkl que tenha um seletor ajustado."""
    pkl_files = sorted(result_dir.rglob("*.pkl"))

    if not pkl_files:
        print(f"[INFO] Nenhum arquivo .pkl encontrado em: {result_dir}")
        return pd.DataFrame()

    print(f"[INFO] {len(pkl_files)} arquivo(s) .pkl encontrado(s) em '{result_dir}'.\n")

    all_rows: list[dict] = []
    for pkl_path in pkl_files:
        rows = extract_rows(pkl_path)
        all_rows.extend(rows)
        label = pkl_path.relative_to(result_dir)
        status = f"{len(rows)} linha(s)" if rows else "sem seletor -- ignorado"
        print(f"  OK  {label}  ->  {status}")

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def aggregate_mean(df_detail: pd.DataFrame) -> pd.DataFrame:
    """Agrega media/desvio de N_Features_Selected por (ExperimentID, Serie,
    Modelo, Strategy) -- N_Features_Total e constante dentro do grupo (mesma
    serie -> mesmo lag_size resolvido), reduzido via 'first'."""
    group_cols = ["ExperimentID", "Serie", "Modelo", "Strategy"]

    df_agg = (
        df_detail
        .groupby(group_cols)["N_Features_Selected"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "N_Features_Selected_mean",
            "std": "N_Features_Selected_std",
            "count": "N_Repeticoes",
        })
    )

    n_total = (
        df_detail
        .groupby(group_cols)["N_Features_Total"]
        .first()
        .reset_index()
    )

    df_agg = df_agg.merge(n_total, on=group_cols, how="left")
    return df_agg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compila o numero/indices de features selecionadas por execucao "
            "de todos os .pkl com seletor em um diretorio de resultados."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"Diretório raiz dos resultados .pkl (padrão: {DEFAULT_RESULT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Caminho do CSV de saída agregado (padrão: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help=(
            "Salva também um CSV com todas as repetições individuais "
            "(salvo ao lado do --output com sufixo _detail)."
        ),
    )
    return parser


def run_export_selected_features(result_dir: Path, output: Path, detail: bool = False) -> pd.DataFrame:
    """
    Nucleo reutilizavel por baixo do `main()` (CLI) e de chamada direta em
    notebook (Tarefa 3.2) -- mesma funcao alimenta os dois pontos de
    entrada, garantindo que o CSV escrito em disco seja identico entre eles
    quando a extracao tem sucesso. Erros (result_dir ausente) se manifestam
    diferente em cada ponto de entrada por design: main() traduz para
    sys.exit(1) com mensagem amigavel; uma chamada direta em notebook deixa
    o FileNotFoundError propagar como traceback normal do Jupyter.

    Ao contrario de `main()`, nunca chama `sys.exit()` -- levanta
    `FileNotFoundError` se `result_dir` nao existir, e retorna um DataFrame
    vazio (sem escrever nada) se nenhum seletor for encontrado, para nao
    interromper um kernel Jupyter.
    """
    if not result_dir.exists():
        raise FileNotFoundError(f"Diretório de resultados não encontrado: {result_dir}")

    df_detail = collect_all_rows(result_dir)

    if df_detail.empty:
        print("[INFO] Nenhum seletor encontrado. Verifique se os .pkl usam Pipeline com step 'selector'.")
        return pd.DataFrame()

    df_mean = aggregate_mean(df_detail)

    save_csv(df_mean, output, label="CSV agregado (média/desvio por série × modelo)")

    if detail:
        detail_path = output.with_name(output.stem + "_detail" + output.suffix)
        save_csv(df_detail, detail_path, label="CSV detalhado (por repetição)")

    return df_mean


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result_dir: Path = args.result_dir.resolve()
    output: Path = args.output.resolve()

    # Verificado aqui (não dentro de um try/except em torno de todo o
    # pipeline) para não confundir um FileNotFoundError não relacionado
    # (ex. vindo de save_csv por outro motivo) com "diretório não existe" --
    # achado de code-review da Tarefa 3.2.
    if not result_dir.exists():
        print(f"[ERRO] Diretório de resultados não encontrado: {result_dir}")
        sys.exit(1)

    df_mean = run_export_selected_features(result_dir, output, detail=args.detail)

    if df_mean.empty:
        sys.exit(0)

    preview_cols = ["ExperimentID", "Serie", "Modelo", "Strategy", "N_Repeticoes",
                     "N_Features_Selected_mean", "N_Features_Selected_std", "N_Features_Total"]
    preview_cols = [c for c in preview_cols if c in df_mean.columns]

    print("\n--- Prévia (primeiras 10 linhas) ---")
    print(df_mean[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
