"""
generate_baseline_report.py
---------------------------
Gera análise completa da baseline a partir dos CSVs de resultados.

Saídas produzidas
-----------------
    results/plots/rmse_comparativo.png      · barras horizontais agrupadas
    results/plots/rmse_estabilidade.png     · boxplot de variância por repetição
    results/baseline_report.md              · relatório Markdown automatizado

Uso
---
    python src/utils/generate_baseline_report.py
    python src/utils/generate_baseline_report.py --results-dir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
ROOT  = _HERE.parents[2]

DEFAULT_RESULTS_DIR  = ROOT / "results"
DEFAULT_PLOTS_DIR    = DEFAULT_RESULTS_DIR / "plots"
DEFAULT_AGG_CSV      = DEFAULT_RESULTS_DIR / "baseline_metrics.csv"
DEFAULT_DETAIL_CSV   = DEFAULT_RESULTS_DIR / "baseline_metrics_detail.csv"
DEFAULT_REPORT_PATH  = DEFAULT_RESULTS_DIR / "baseline_report.md"

# ---------------------------------------------------------------------------
# Mapeamento de nomes internos → rótulos legíveis
# ---------------------------------------------------------------------------
MODEL_LABELS: dict[str, str] = {
    "arima":  "ARIMA",
    "mlp":    "MLP",
    "svr":    "SVR",
    "amv1":   "ARIMA-MLP",
    "as":     "ARIMA-SVR",
}

# Ordem de exibição nos gráficos
MODEL_ORDER = ["arima", "mlp", "svr", "amv1", "as"]

# Paleta de cores: uma cor distinta por modelo
PALETTE = {
    "arima":  "#4C72B0",
    "mlp":    "#DD8452",
    "svr":    "#55A868",
    "amv1":   "#C44E52",
    "as":     "#8172B2",
}

# ---------------------------------------------------------------------------
# 1. Carga e limpeza dos dados
# ---------------------------------------------------------------------------

def load_agg(path: Path) -> pd.DataFrame:
    """
    Carrega baseline_metrics.csv, limpa o prefixo de horizonte do nome do
    modelo e retorna DataFrame com coluna `Modelo` normalizada.

    O `format_names` de generics.py gera nomes como ``airlines_1mlp.pkl``.
    Após o parse do export_metrics, a coluna Modelo contém "1mlp", "1arima",
    etc. Este prefixo "1" (horizonte) é removido aqui para facilitar a análise.
    """
    df = pd.read_csv(path)

    # Remove prefixo numérico do modelo: "1mlp" → "mlp", "1amv1" → "amv1"
    df["Modelo"] = df["Modelo"].str.lstrip("0123456789")

    # Mantém apenas modelos conhecidos
    df = df[df["Modelo"].isin(MODEL_LABELS)].copy()

    return df


def load_detail(path: Path) -> pd.DataFrame:
    """
    Carrega baseline_metrics_detail.csv com o mesmo tratamento de nomes.
    Filtra apenas modelos estocásticos (MLP e híbridos) que têm repetições.
    """
    df = pd.read_csv(path)
    df["Modelo"] = df["Modelo"].str.lstrip("0123456789")
    df = df[df["Modelo"].isin(MODEL_LABELS)].copy()

    # Modelos com repetições para análise de estabilidade
    stochastic = ["mlp", "amv1"]
    df = df[df["Modelo"].isin(stochastic)].copy()

    return df


# ---------------------------------------------------------------------------
# 2. Tabela pivô de RMSE
# ---------------------------------------------------------------------------

def make_rmse_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DataFrame pivô: índice=Serie, colunas=Modelo, valores=RMSE_mean.
    Ordem de colunas segue MODEL_ORDER.
    """
    pivot = df.pivot_table(
        index="Serie", columns="Modelo", values="RMSE_mean", aggfunc="first"
    )

    # Reordena colunas para os modelos disponíveis
    cols = [m for m in MODEL_ORDER if m in pivot.columns]
    pivot = pivot[cols]

    # Renomeia colunas para rótulos legíveis
    pivot.columns = [MODEL_LABELS[c] for c in pivot.columns]

    return pivot.sort_index()


# ---------------------------------------------------------------------------
# 3. Ganho Híbrido
# ---------------------------------------------------------------------------

def make_hybrid_gain(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a redução percentual de RMSE dos modelos híbridos sobre o ARIMA.

    Ganho > 0  → híbrido é melhor que ARIMA (erro menor)
    Ganho < 0  → híbrido é pior que ARIMA

    Fórmula: (RMSE_ARIMA - RMSE_Hibrido) / RMSE_ARIMA × 100
    """
    gain = pd.DataFrame(index=pivot.index)

    arima_col = MODEL_LABELS["arima"]   # "ARIMA"

    for hybrid_key in ["amv1", "as"]:
        col_label = MODEL_LABELS[hybrid_key]
        if arima_col in pivot.columns and col_label in pivot.columns:
            gain[f"Ganho {col_label} vs ARIMA (%)"] = (
                (pivot[arima_col] - pivot[col_label]) / pivot[arima_col] * 100
            ).round(2)

    return gain


# ---------------------------------------------------------------------------
# 4. Gráficos
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    })


def plot_rmse_grouped_bars(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Barras horizontais agrupadas por modelo para cada série.
    Dividido em dois painéis (9 + 8 séries) para manter legibilidade.
    """
    _setup_style()

    series_list = sorted(df["Serie"].unique())
    n_half      = (len(series_list) + 1) // 2
    groups      = [series_list[:n_half], series_list[n_half:]]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("RMSE Médio por Modelo e Série — Baseline Completa", fontsize=14, y=1.01)

    models_in_data = [m for m in MODEL_ORDER if m in df["Modelo"].unique()]
    bar_h = 0.14
    offsets = np.linspace(
        -(len(models_in_data) - 1) * bar_h / 2,
         (len(models_in_data) - 1) * bar_h / 2,
        len(models_in_data),
    )

    for ax, group in zip(axes, groups):
        group_df = df[df["Serie"].isin(group)].copy()
        y_pos    = np.arange(len(group))

        for i, model_key in enumerate(models_in_data):
            subset = group_df[group_df["Modelo"] == model_key]
            # alinha valores pela ordem de `group`
            rmse_vals = [
                subset.loc[subset["Serie"] == s, "RMSE_mean"].values[0]
                if s in subset["Serie"].values else np.nan
                for s in group
            ]
            ax.barh(
                y_pos + offsets[i],
                rmse_vals,
                height=bar_h,
                label=MODEL_LABELS[model_key],
                color=PALETTE[model_key],
                alpha=0.88,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(group, fontsize=10)
        ax.set_xlabel("RMSE Médio", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.4)

    # Legenda única acima da figura
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(models_in_data),
        bbox_to_anchor=(0.5, 1.04), frameon=True, fontsize=11,
    )

    plt.tight_layout()
    out = output_dir / "rmse_comparativo.png"
    plt.savefig(out)
    plt.close()
    return out


def plot_stability_boxplot(df_detail: pd.DataFrame, output_dir: Path) -> Path:
    """
    Boxplot da distribuição de RMSE por repetição para MLP e ARIMA-MLP,
    agrupado por série. Mostra estabilidade entre execuções estocásticas.
    """
    _setup_style()

    if df_detail.empty:
        # Se não há dados de repetições, cria figura placeholder
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sem dados de repetições disponíveis",
                ha="center", va="center", transform=ax.transAxes)
        out = output_dir / "rmse_estabilidade.png"
        plt.savefig(out)
        plt.close()
        return out

    # Normaliza escala: log(RMSE) para comparar séries de magnitudes diferentes
    df_detail = df_detail.copy()
    df_detail["log_RMSE"] = np.log1p(df_detail["RMSE"])
    df_detail["Modelo_label"] = df_detail["Modelo"].map(MODEL_LABELS)

    series_list = sorted(df_detail["Serie"].unique())
    n_series    = len(series_list)

    n_cols = 4
    n_rows = int(np.ceil(n_series / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.5, n_rows * 3.5),
        sharey=False,
    )
    axes_flat = axes.flatten() if n_series > 1 else [axes]

    fig.suptitle(
        "Distribuição de RMSE por Repetição (MLP e ARIMA-MLP)\n"
        "Eixo Y: log(1 + RMSE) para escala comparável",
        fontsize=13, y=1.01,
    )

    model_palette = {MODEL_LABELS[m]: PALETTE[m] for m in ["mlp", "amv1"]}

    for idx, serie in enumerate(series_list):
        ax  = axes_flat[idx]
        sub = df_detail[df_detail["Serie"] == serie]

        sns.boxplot(
            data=sub,
            x="Modelo_label",
            y="log_RMSE",
            hue="Modelo_label",
            palette=model_palette,
            legend=False,
            width=0.5,
            linewidth=1.2,
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
            ax=ax,
        )
        ax.set_title(serie, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("log(1+RMSE)" if idx % n_cols == 0 else "")
        ax.grid(axis="y", alpha=0.4)
        ax.tick_params(axis="x", labelsize=9)

    # Oculta eixos extras
    for ax in axes_flat[n_series:]:
        ax.set_visible(False)

    plt.tight_layout()
    out = output_dir / "rmse_estabilidade.png"
    plt.savefig(out)
    plt.close()
    return out


# ---------------------------------------------------------------------------
# 5. Relatório Markdown
# ---------------------------------------------------------------------------

def _df_to_md(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Converte DataFrame para tabela Markdown alinhada."""
    col_header = "| " + " | ".join(str(c) for c in df.reset_index().columns) + " |"
    separator  = "| " + " | ".join(["---"] * len(df.reset_index().columns)) + " |"

    rows = []
    for _, row in df.reset_index().iterrows():
        parts = []
        for val in row:
            if isinstance(val, float):
                parts.append(f"{val:{float_fmt}}")
            else:
                parts.append(str(val))
        rows.append("| " + " | ".join(parts) + " |")

    return "\n".join([col_header, separator] + rows)


def generate_report(
    df_agg:    pd.DataFrame,
    df_detail: pd.DataFrame,
    pivot:     pd.DataFrame,
    gain:      pd.DataFrame,
    plots_dir: Path,
    output:    Path,
) -> Path:
    """Gera o relatório Markdown completo."""

    # --- Modelo vencedor (menor RMSE médio por série) ---
    models_in_data = [m for m in MODEL_ORDER if m in df_agg["Modelo"].unique()]
    winner_per_serie = (
        df_agg.groupby(["Serie", "Modelo"])["RMSE_mean"]
        .first()
        .reset_index()
        .sort_values("RMSE_mean")
        .drop_duplicates("Serie")
        .groupby("Modelo")
        .size()
        .sort_values(ascending=False)
    )
    top_model_key   = winner_per_serie.index[0]
    top_model_label = MODEL_LABELS.get(top_model_key, top_model_key)
    top_model_wins  = int(winner_per_serie.iloc[0])
    n_series        = df_agg["Serie"].nunique()

    # --- Média global de RMSE por modelo ---
    global_mean = (
        df_agg.groupby("Modelo")["RMSE_mean"]
        .mean()
        .reindex([m for m in MODEL_ORDER if m in df_agg["Modelo"].unique()])
        .rename(index=MODEL_LABELS)
        .reset_index()
        .rename(columns={"index": "Modelo", "RMSE_mean": "RMSE_Médio_Global"})
    )
    global_mean["RMSE_Médio_Global"] = global_mean["RMSE_Médio_Global"].round(4)

    # --- Ganho híbrido: estatísticas resumidas ---
    if not gain.empty:
        gain_summary = gain.agg(["mean", "min", "max"]).T.round(2)
        gain_summary.columns = ["Média (%)", "Mín (%)", "Máx (%)"]

    # --- Pivot formatado (4 casas decimais) ---
    pivot_display = pivot.round(4)

    # --- Relative path para imagens (relativa a results/) ---
    rel_plots = Path("plots")

    # -----------------------------------------------------------------------
    lines: list[str] = []

    lines += [
        "# Relatório de Baseline — Previsão de Séries Temporais",
        "",
        f"> Gerado automaticamente por `generate_baseline_report.py`  ",
        f"> **{n_series} séries · {len(models_in_data)} modelos · experimento `chamados`**",
        "",
        "---",
        "",
        "## 1. Resumo Executivo",
        "",
        f"O modelo **{top_model_label}** obteve o menor RMSE em **{top_model_wins} de {n_series} séries** "
        f"considerando os valores médios das métricas de teste.",
        "",
    ]

    # Tabela de vitórias por modelo
    lines += [
        "### Séries vencidas por modelo (menor RMSE médio no conjunto de teste)",
        "",
    ]
    win_rows = winner_per_serie.reset_index()
    win_rows.columns = ["Modelo", "Séries Vencidas"]
    win_rows["Modelo"] = win_rows["Modelo"].map(lambda m: MODEL_LABELS.get(m, m))
    lines.append(_df_to_md(win_rows.set_index("Modelo"), float_fmt=".0f"))
    lines.append("")

    # -----------------------------------------------------------------------
    lines += [
        "---",
        "",
        "## 2. Tabela Comparativa de RMSE Médio",
        "",
        "_Linhas: séries temporais. Colunas: modelos. Valores: RMSE médio no conjunto de teste._",
        "_Menores valores são melhores._",
        "",
        _df_to_md(pivot_display),
        "",
    ]

    # -----------------------------------------------------------------------
    lines += [
        "---",
        "",
        "## 3. Ganho Híbrido sobre o ARIMA",
        "",
        "Redução percentual de RMSE dos modelos híbridos em relação ao ARIMA puro.",
        "",
        "$$\\text{Ganho} = \\frac{\\text{RMSE}_{\\text{ARIMA}} - \\text{RMSE}_{\\text{Híbrido}}"
        "}{\\text{RMSE}_{\\text{ARIMA}}} \\times 100$$",
        "",
        "> **Positivo** → híbrido melhora o ARIMA  |  **Negativo** → híbrido piora o ARIMA",
        "",
    ]

    if not gain.empty:
        lines.append(_df_to_md(gain.round(2)))
        lines.append("")
        lines += [
            "### 3.1 Estatísticas do Ganho Híbrido",
            "",
            _df_to_md(gain_summary),
            "",
        ]
    else:
        lines.append("_Dados de ganho não disponíveis._\n")

    # -----------------------------------------------------------------------
    lines += [
        "---",
        "",
        "## 4. Média Global de RMSE por Modelo",
        "",
        "_Média do RMSE médio cruzando todas as séries da baseline._",
        "",
        _df_to_md(global_mean.set_index("Modelo"), float_fmt=".4f"),
        "",
    ]

    # -----------------------------------------------------------------------
    lines += [
        "---",
        "",
        "## 5. Visualizações",
        "",
        "### 5.1 RMSE Comparativo por Série",
        "",
        f"![RMSE comparativo]({rel_plots / 'rmse_comparativo.png'})",
        "",
        "### 5.2 Estabilidade das Repetições (MLP e ARIMA-MLP)",
        "",
        "_Distribuição do RMSE ao longo das 10 repetições independentes._  ",
        "_Eixo Y em escala log(1 + RMSE) para comparabilidade entre séries de magnitudes distintas._",
        "",
        f"![Estabilidade]({rel_plots / 'rmse_estabilidade.png'})",
        "",
        "---",
        "",
        "## 6. Observações Metodológicas",
        "",
        "- **Modelo ARIMA** e **SVR**: executados com 1 repetição (determinísticos / sem aleatoriedade configurada).",
        "- **MLP** e **ARIMA-MLP**: 10 repetições independentes com pesos aleatórios — médias e desvios reportados.",
        "- Métricas calculadas exclusivamente no **conjunto de teste** (10% finais de cada série).",
        "- Normalização Min-Max aplicada apenas sobre o conjunto de treino (sem *data leakage*).",
        "- Diferenciação KPSS ativada quando a série não é estacionária.",
        "",
    ]

    report = "\n".join(lines)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera relatório completo da baseline (gráficos + Markdown)."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    results_dir = args.results_dir.resolve()
    agg_csv     = results_dir / "baseline_metrics.csv"
    detail_csv  = results_dir / "baseline_metrics_detail.csv"
    plots_dir   = results_dir / "plots"
    report_path = results_dir / "baseline_report.md"

    # --- validações ---
    if not agg_csv.exists():
        print(f"[ERRO] Arquivo não encontrado: {agg_csv}")
        sys.exit(1)

    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- carga ---
    print("[1/5] Carregando dados...")
    df_agg    = load_agg(agg_csv)
    df_detail = load_detail(detail_csv) if detail_csv.exists() else pd.DataFrame()

    if df_agg.empty:
        print("[ERRO] Nenhum dado reconhecido no CSV. Verifique os nomes dos modelos.")
        sys.exit(1)

    # --- processamento ---
    print("[2/5] Calculando tabela pivô e ganho híbrido...")
    pivot = make_rmse_pivot(df_agg)
    gain  = make_hybrid_gain(pivot)

    # --- gráficos ---
    print("[3/5] Gerando gráfico de barras comparativo...")
    p1 = plot_rmse_grouped_bars(df_agg, plots_dir)
    print(f"      -> {p1.relative_to(ROOT) if p1.is_absolute() else p1}")

    print("[4/5] Gerando boxplot de estabilidade...")
    p2 = plot_stability_boxplot(df_detail, plots_dir)
    print(f"      -> {p2.relative_to(ROOT) if p2.is_absolute() else p2}")

    # --- relatório ---
    print("[5/5] Escrevendo baseline_report.md...")
    out = generate_report(df_agg, df_detail, pivot, gain, plots_dir, report_path)
    print(f"      -> {out.relative_to(ROOT) if out.is_absolute() else out}")

    # --- preview no terminal ---
    print("\n--- Tabela Pivô de RMSE (prévia) ---")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", 10):
        print(pivot.to_string())

    if not gain.empty:
        print("\n--- Ganho Híbrido (%) ---")
        with pd.option_context("display.float_format", "{:.2f}".format):
            print(gain.to_string())

    print("\n[OK] Relatório completo gerado em:", report_path)


if __name__ == "__main__":
    main()
