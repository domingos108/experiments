"""
build_tabela_orientador.py
---------------------------
Gera results/tabela_consolidada_orientador.md a partir de
results/benchmark_master_with_sources_v1.csv (Tarefa 9) -- formaliza como
script reutilizavel a tabela entregue manualmente ao orientador (Tarefa 14),
para que possa ser regenerada sempre que a tabela mestre ganhar novos
experimentos, sem re-hardcodar numeros.

Escopo da tabela (decisao do orientador): ARIMA/MLP/SVR aparecem so como
referencia (linha 'sem_FS', sem Feature Selection -- FS nao e aplicado a
modelos single isolados neste documento, ainda que a tabela mestre tenha
essas linhas de FS disponiveis para outros usos). ARIMA-MLP/ARIMA-SVR (via
Additive) trazem o baseline sem FS e as 5 variantes de FS. Menor RMSE de
cada serie em negrito (todos os empates, nao so o primeiro).

Uso
---
    python src/utils/build_tabela_orientador.py \
        --csv results/benchmark_master_with_sources_v1.csv \
        --output results/tabela_consolidada_orientador.md
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

DEFAULT_CSV = ROOT / "results" / "benchmark_master_with_sources_v1.csv"
DEFAULT_OUTPUT = ROOT / "results" / "tabela_consolidada_orientador.md"

# Familias que aparecem so como referencia (linha 'sem_FS'), nunca com as
# variantes de FS, mesmo quando a tabela mestre (Tarefa 9) tem essas linhas
# disponiveis (SVR/MLP single tambem rodaram FS, ver Tarefa 5/6) -- decisao
# de escopo do orientador para ESTE documento, nao uma limitacao dos dados.
FAMILY_REFERENCE = ["ARIMA", "MLP", "SVR"]

# Familias hibridas: baseline sem FS + as 5 variantes, sempre nesta ordem
# (nao a ordem incidental de linhas do CSV de origem).
FAMILY_HYBRID = ["ARIMA-MLP", "ARIMA-SVR"]

METHOD_ORDER = ["sem_FS", "ftest", "mutualinfo", "rfembedded", "lasso", "rfecv"]

METHOD_DISPLAY = {
    "sem_FS": "sem Feature Selection",
    "ftest": "f_test",
    "mutualinfo": "mutual_info",
    "rfembedded": "rf_embedded",
    "lasso": "lasso",
    "rfecv": "rfecv",
}

SERIES_ORDER = ["airlines", "austres", "coloradoRiver", "sunspot"]

# Cabecalho fixo do documento (Tarefa 14) -- a nota metodologica sobre k>=N
# vem da investigacao caso a caso da Tarefa 13 (load_grid_search_history() +
# codigo-fonte dos notebooks), nao e recalculada aqui.
HEADER = """# Tabela Consolidada — ARIMA, MLP, SVR (referência) x ARIMA-MLP, ARIMA-SVR (com FS)

Solicitada pelo orientador. Escopo: `ARIMA`, `MLP`, `SVR` aparecem apenas como referência (sem Feature Selection, por decisão do orientador — FS não é aplicado a modelos single isolados). `ARIMA-MLP` e `ARIMA-SVR` (via Additive) trazem o baseline sem FS e as 5 variantes de FS cada. Melhor RMSE de cada série em **negrito**. Fonte: `results/benchmark_master_with_sources_v1.csv` (Tarefa 9), validado pela Tarefa 13 (nenhum bug de configuração encontrado nos casos de empate).

> **Nota metodológica**: em `f_test`/`mutual_info`, quando o `k` vencedor do grid é maior ou igual ao número total de lags candidatos da série, `SelectKBest` se torna uma transformação identidade (usa todas as features) — o resultado idêntico ao baseline nesses casos é uma consequência matemática da definição de top-k, não uma falha do seletor. Confirmado com histórico completo de validação por `k` (Tarefa 13)."""


def select_table_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra e ordena as linhas da tabela final: familias de referencia
    (FAMILY_REFERENCE) mantem so a linha 'sem_FS'; familias hibridas
    (FAMILY_HYBRID) mantem as 6 linhas (sem_FS + 5 metodos), sempre na ordem
    de METHOD_ORDER -- nunca a ordem incidental de `df`.

    Espera `df` ja filtrado por serie (uma chamada por serie) OU com as
    series ja na ordem desejada -- esta funcao preserva a ordem de entrada
    de `Serie`, so reordena Familia/Metodo_FS dentro dela.
    """
    rows = []
    for serie in df["Serie"].drop_duplicates():
        serie_df = df[df["Serie"] == serie]
        for familia in FAMILY_REFERENCE:
            rows.append(serie_df[(serie_df["Familia"] == familia) & (serie_df["Metodo_FS"] == "sem_FS")])
        for familia in FAMILY_HYBRID:
            fam_df = serie_df[serie_df["Familia"] == familia]
            for metodo in METHOD_ORDER:
                rows.append(fam_df[fam_df["Metodo_FS"] == metodo])

    return pd.concat(rows, ignore_index=True) if rows else df.iloc[0:0]


def mark_bold(df: pd.DataFrame, value_col: str = "RMSE", group_col: str = "Serie", decimals: int = 6) -> pd.DataFrame:
    """
    Adiciona a coluna booleana 'Bold': True em TODA linha cujo `value_col`
    empata no minimo do grupo `group_col` (arredondado a `decimals` casas
    para nao perder empates genuinos por ruido de ponto flutuante -- ex.
    ARIMA-SVR/airlines tem 3 linhas com o MESMO valor de RMSE, Tarefa 13).
    """
    out = df.copy()
    rounded = out[value_col].round(decimals)
    group_min = rounded.groupby(out[group_col]).transform("min")
    out["Bold"] = rounded == group_min
    return out


def _format_metric(value, decimals: int = 4) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}"


def _format_n_features(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.1f}"


def _serie_heading(serie: str, serie_df: pd.DataFrame) -> str:
    heading = f"## {serie}"
    if not bool(serie_df["Trivial"].iloc[0]):
        return heading
    n_lags = serie_df["NFeatures"].dropna()
    if n_lags.empty:
        return heading
    n = int(n_lags.iloc[0])
    plural = "s" if n != 1 else ""
    return f"{heading} (série trivial — {n} lag{plural} candidato{plural})"


def render_markdown(df: pd.DataFrame) -> str:
    """Renderiza o documento completo: cabecalho fixo (HEADER) + uma secao
    '## <serie>' por serie de SERIES_ORDER, cada uma com sua tabela."""
    lines = [HEADER, ""]

    for serie in SERIES_ORDER:
        serie_df = df[df["Serie"] == serie]
        if serie_df.empty:
            continue

        lines.append("")
        lines.append(_serie_heading(serie, serie_df))
        lines.append("")
        lines.append("| Família | Método | RMSE | MAPE | NFeatures | Fonte |")
        lines.append("|---|---|---|---|---|---|")

        for _, row in serie_df.iterrows():
            rmse_str = _format_metric(row["RMSE"])
            if row["Bold"]:
                rmse_str = f"**{rmse_str}**"
            mape_str = _format_metric(row["MAPE"])
            nfeat_str = _format_n_features(row["NFeatures"])
            metodo_label = METHOD_DISPLAY.get(row["Metodo_FS"], row["Metodo_FS"])
            lines.append(f"| {row['Familia']} | {metodo_label} | {rmse_str} | {mape_str} | {nfeat_str} | `{row['Fonte']}` |")

    return "\n".join(lines) + "\n"


def build_tabela_orientador(csv_path: Path = DEFAULT_CSV, output_path: Path = DEFAULT_OUTPUT) -> str:
    """Nucleo reutilizavel: le o CSV mestre, seleciona/ordena as linhas do
    escopo desta tabela, marca negrito e escreve o markdown em disco.
    Retorna o texto gerado (mesmo padrao de outros utilitarios do projeto,
    para reuso direto em notebook sem re-parsear o arquivo escrito)."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "RMSE_FS": "RMSE",
        "MAPE_FS": "MAPE",
        "Metrics_Source_File": "Fonte",
    })

    df = select_table_rows(df)
    df = mark_bold(df)
    markdown = render_markdown(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    return markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera results/tabela_consolidada_orientador.md a partir da tabela mestre (Tarefa 9)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"CSV mestre de entrada (padrão: {DEFAULT_CSV})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Caminho do .md de saída (padrão: {DEFAULT_OUTPUT})")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    markdown = build_tabela_orientador(args.csv.resolve(), args.output.resolve())
    print(f"[OK] Tabela escrita em: {args.output.resolve()}")
    print(f"[OK] {markdown.count(chr(10))} linhas geradas.")


if __name__ == "__main__":
    main()
