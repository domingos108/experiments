"""
export_metrics_to_csv.py
------------------------
Varre data/result/, lê todos os .pkl gerados pelo pipeline de experimentos e
compila as métricas de teste (test_metrics) em results/baseline_metrics.csv.

Uso
---
    # Dentro do ambiente virtual do projeto:
    python src/utils/export_metrics_to_csv.py

    # Com opções:
    python src/utils/export_metrics_to_csv.py --result-dir data/result/ \
                                               --output    results/baseline_metrics.csv \
                                               --detail

Argumentos opcionais
--------------------
--result-dir  Diretório raiz onde estão os .pkl (padrão: config.MODEL_DATA_PATH).
--output      Caminho do CSV de saída                (padrão: results/baseline_metrics.csv).
--detail      Se presente, salva TAMBÉM um CSV com todas as repetições individuais
              (results/baseline_metrics_detail.csv), sem agregar por média.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolução de paths — funciona tanto ao rodar diretamente quanto via import
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]          # <repo>/
SRC  = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config 

from model.generics import ResultExp 

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEFAULT_RESULT_DIR = Path(config.MODEL_DATA_PATH)
DEFAULT_OUTPUT     = ROOT / "results" / "baseline_metrics.csv"

# Chaves de métricas presentes em test_metrics / val_metrics
METRIC_KEYS: list[str] = ["MSE", "RMSE", "MAE", "MAPE", "theil", "ARV", "IA", "POCID"]


# ---------------------------------------------------------------------------
# Parsing do caminho
# ---------------------------------------------------------------------------

def parse_pkl_path(pkl_path: Path) -> tuple[str, str, str]:
    """
    Extrai (experiment_id, serie, modelo) a partir do caminho do .pkl.

    Estrutura esperada pelo ``generics.format_names``:
        <MODEL_DATA_PATH>/<experiment_id>/<serie>_<modelo>.pkl

    A mesma lógica de ``metrics.open_fold_result`` é usada:
    - ``modelo`` → último token depois do último '_'
    - ``serie``  → tudo antes do último '_'

    Exemplos
    --------
    parse_pkl_path(Path("data/result/chamados/airlines_mlp.pkl"))
    ('chamados', 'airlines', 'mlp')
    parse_pkl_path(Path("data/result/exp1/windspeedrecife_svr.pkl"))
    ('exp1', 'windspeedrecife', 'svr')
    """
    experiment_id = pkl_path.parent.name
    stem = pkl_path.stem                        

    # rsplit com maxsplit=1 → separa sempre no ÚLTIMO underscore
    parts = stem.rsplit("_", maxsplit=1)
    if len(parts) == 2:
        serie, modelo = parts
    else:
        # arquivo sem underscore no nome — guarda tudo como serie
        serie  = stem
        modelo = "unknown"

    return experiment_id, serie, modelo


# ---------------------------------------------------------------------------
# Leitura e extração de um único .pkl
# ---------------------------------------------------------------------------

def _load_pkl(path: Path) -> list:
    """Desserializa o .pkl e garante retorno como lista."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data if isinstance(data, list) else [data]


def _unwrap_entry(entry):
    """
    Suporte a dois formatos possíveis de item dentro do .pkl:
        1) dict {'experiment': ResultExp/Additive/SKlearnModel, 'val_metric': float}  ← padrão GridSearch
        2) ResultExp/Additive/SKlearnModel diretamente                                  ← salvamento simples

    Retorna (result_exp, val_metric) -- val_metric é None quando o formato 2 é usado.
    Compartilhado por export_metrics_to_csv.py e export_selected_features.py, que leem
    os mesmos .pkl com extrações diferentes.
    """
    if isinstance(entry, dict) and "experiment" in entry:
        return entry["experiment"], entry.get("val_metric")
    return entry, None


def extract_rows(pkl_path: Path) -> list[dict]:
    """
    Lê um .pkl e retorna uma lista de dicionários (uma linha por repetição).

    Cada dicionário contém:
        ExperimentID, Serie, Modelo, Repeticao, ValMetric,
        MSE, RMSE, MAE, MAPE, theil, ARV, IA, POCID

    O campo ``ValMetric`` corresponde ao melhor val_metric registrado durante a
    GridSearch que gerou aquela repetição (pode ser NaN se não disponível).
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
            result_exp, val_metric = _unwrap_entry(entry)

            test_metrics: dict = result_exp.metrics_results.get("test_metrics", {})

            row: dict = {
                "ExperimentID": experiment_id,
                "Serie":        serie,
                "Modelo":       modelo,
                "Repeticao":    repetition_idx,
                "ValMetric":    val_metric if val_metric is not None else np.nan,
            }

            for key in METRIC_KEYS:
                row[key] = test_metrics.get(key, np.nan)

            rows.append(row)

        except Exception as exc:
            print(
                f"  [AVISO] Erro na repetição {repetition_idx} "
                f"de {pkl_path.name}: {exc}"
            )

    return rows


# ---------------------------------------------------------------------------
# Varredura do diretório e compilação do DataFrame
# ---------------------------------------------------------------------------

def collect_all_rows(result_dir: Path) -> pd.DataFrame:
    """
    Varre ``result_dir`` recursivamente, extrai métricas de cada .pkl e
    retorna um DataFrame com todas as linhas (uma por repetição por arquivo).
    """
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
        status = f"{len(rows)} linha(s)" if rows else "ignorado"
        print(f"  OK  {label}  ->  {status}")

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Agregação: média das repetições
# ---------------------------------------------------------------------------

def aggregate_mean(df_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega o DataFrame de detalhes calculando a média de cada métrica por
    (ExperimentID, Serie, Modelo).

    Também adiciona a coluna ``N_Repeticoes`` indicando quantas execuções
    foram consideradas na média.
    """
    group_cols = ["ExperimentID", "Serie", "Modelo"]

    df_agg = (
        df_detail
        .groupby(group_cols)[METRIC_KEYS]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Achatar MultiIndex de colunas: ('RMSE', 'mean') → 'RMSE_mean'
    df_agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in df_agg.columns
    ]

    # Contagem de repetições válidas (usa RMSE como referência)
    n_reps = (
        df_detail
        .groupby(group_cols)["RMSE"]
        .count()
        .reset_index(name="N_Repeticoes")
    )

    df_agg = df_agg.merge(n_reps, on=group_cols, how="left")

    # Reordenar: identificadores primeiro, depois as métricas ordenadas
    id_cols  = group_cols + ["N_Repeticoes"]
    met_cols = [c for c in df_agg.columns if c not in id_cols]
    df_agg   = df_agg[id_cols + met_cols]

    return df_agg


# ---------------------------------------------------------------------------
# Escrita do CSV
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, output_path: Path, label: str = "CSV") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\n[OK] {label} gerado em: {output_path}")
    print(f"     {len(df)} linha(s) × {len(df.columns)} coluna(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compila as métricas de teste de todos os .pkl em data/result/ "
            "e salva em um único CSV."
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
            "Salva também um CSV com todos os valores por repetição "
            "(salvo ao lado do --output com sufixo _detail)."
        ),
    )
    return parser


def run_export_metrics_to_csv(result_dir: Path, output: Path, detail: bool = False) -> pd.DataFrame:
    """
    Nucleo reutilizavel por baixo do `main()` (CLI) e de chamada direta em
    notebook (Tarefa 3.2) -- a mesma funcao alimenta os dois pontos de
    entrada, garantindo que o CSV escrito em disco seja identico entre eles
    quando a extracao tem sucesso. Erros (result_dir ausente) se manifestam
    diferente em cada ponto de entrada por design: main() traduz para
    sys.exit(1) com mensagem amigavel; uma chamada direta em notebook deixa
    o FileNotFoundError propagar como traceback normal do Jupyter.

    Ao contrario de `main()`, nunca chama `sys.exit()` -- levanta
    `FileNotFoundError` se `result_dir` nao existir, e retorna um DataFrame
    vazio (sem escrever nada) se nenhuma metrica for extraida, para nao
    interromper um kernel Jupyter.
    """
    if not result_dir.exists():
        raise FileNotFoundError(f"Diretório de resultados não encontrado: {result_dir}")

    df_detail = collect_all_rows(result_dir)

    if df_detail.empty:
        print("[INFO] Nenhuma métrica extraída. Verifique se os .pkl foram gerados.")
        return pd.DataFrame()

    df_mean = aggregate_mean(df_detail)

    save_csv(df_mean, output, label="CSV agregado (média das repetições)")

    if detail:
        detail_path = output.with_name(output.stem + "_detail" + output.suffix)
        save_csv(df_detail, detail_path, label="CSV detalhado (por repetição)")

    return df_mean


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    result_dir: Path = args.result_dir.resolve()
    output:     Path = args.output.resolve()

    # Verificado aqui (não dentro de um try/except em torno de todo o
    # pipeline) para não confundir um FileNotFoundError não relacionado
    # (ex. vindo de save_csv por outro motivo) com "diretório não existe" --
    # achado de code-review da Tarefa 3.2.
    if not result_dir.exists():
        print(f"[ERRO] Diretório de resultados não encontrado: {result_dir}")
        sys.exit(1)

    df_mean = run_export_metrics_to_csv(result_dir, output, detail=args.detail)

    if df_mean.empty:
        sys.exit(0)

    # ---- Prévia no terminal ----
    preview_cols = ["ExperimentID", "Serie", "Modelo", "N_Repeticoes",
                    "RMSE_mean", "MAE_mean", "MAPE_mean"]
    preview_cols = [c for c in preview_cols if c in df_mean.columns]

    print("\n--- Prévia (primeiras 10 linhas) ---")
    print(df_mean[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
