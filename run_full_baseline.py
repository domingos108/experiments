"""
run_full_baseline.py
--------------------
Orquestrador da baseline.

Executa em sequência os notebooks de experimento usando papermill e,
ao final, compila todas as métricas em results/baseline_metrics.csv.

Uso
---
    python run_full_baseline.py               # roda tudo
    python run_full_baseline.py --dry-run     # apenas lista o que seria executado
    python run_full_baseline.py --from arima_exec  # retoma a partir de um notebook
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolução da raiz do projeto
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Pipeline de notebooks (ARIMA deve preceder os híbridos)
# ---------------------------------------------------------------------------
PIPELINE: list[dict] = [
    {
        "label":    "1/5 · ARIMA (modelo linear base)",
        "notebook": ROOT / "notebook" / "single_models"          / "arima_exec.ipynb",
    },
    {
        "label":    "2/5 · MLP  (modelo não-linear)",
        "notebook": ROOT / "notebook" / "single_models"          / "mlp_exec.ipynb",
    },
    {
        "label":    "3/5 · SVR  (modelo não-linear)",
        "notebook": ROOT / "notebook" / "single_models"          / "svr_exec.ipynb",
    },
    {
        "label":    "4/5 · ARIMA-MLP (híbrido aditivo)",
        "notebook": ROOT / "notebook" / "residual_hybridsystem"  / "arima_mlp.ipynb",
    },
    {
        "label":    "5/5 · ARIMA-SVR (híbrido aditivo)",
        "notebook": ROOT / "notebook" / "residual_hybridsystem"  / "arima_svr.ipynb",
    },
]

OUTPUT_BASE = ROOT / "notebook" / "executed"   # notebooks executados (com outputs)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check_papermill() -> None:
    """Verifica se papermill está instalado e aborta se não."""
    try:
        import papermill  # noqa: F401
    except ImportError:
        _log(
            "\n[ERRO] papermill não está instalado.\n"
            "       Execute:  pip install papermill\n"
            "       Ou:       pip install -r requirements.txt\n"
        )
        sys.exit(1)


def _build_env() -> dict:
    """
    Cria uma cópia do ambiente do processo atual com ROOT e SRC
    adicionados ao início de PYTHONPATH.

    Isso garante que os notebooks executados pelo papermill possam
    resolver ``from model import generics``, ``import config`` etc.,
    exatamente como fazem quando rodados manualmente no Jupyter.
    """
    env = os.environ.copy()
    pythonpath_extra = os.pathsep.join([str(SRC), str(ROOT)])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{pythonpath_extra}{os.pathsep}{existing}" if existing else pythonpath_extra
    )
    return env


def _resolve_start_index(from_label: str | None) -> int:
    """Retorna o índice do pipeline a partir do qual retomar a execução."""
    if from_label is None:
        return 0

    needle = from_label.lower()
    for idx, step in enumerate(PIPELINE):
        nb_stem = step["notebook"].stem.lower()
        if needle in nb_stem or needle in step["label"].lower():
            return idx

    valid = ", ".join(s["notebook"].stem for s in PIPELINE)
    _log(f"[ERRO] Notebook '{from_label}' não encontrado no pipeline.\nVálidos: {valid}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Execução de um único notebook via papermill
# ---------------------------------------------------------------------------

def run_notebook(step: dict, output_dir: Path, dry_run: bool) -> bool:
    """
    Executa o notebook via ``python -m papermill`` como subprocess.

    PYTHONPATH injetado: ROOT/src + ROOT  → cobre ``import config``,
    ``from model import ...``, ``from input import ...``, etc.

    Retorna True em caso de sucesso, False em caso de falha.
    """
    nb_path  = step["notebook"]
    out_path = output_dir / nb_path.name

    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"\n{'='*60}")
    _log(f"  {step['label']}")
    _log(f"  Notebook : {nb_path.relative_to(ROOT)}")
    _log(f"  Saída    : {out_path.relative_to(ROOT)}")
    _log(f"{'='*60}")

    if dry_run:
        _log("  [DRY-RUN] Pulando execução real.")
        return True

    cmd = [
        sys.executable, "-m", "papermill",
        str(nb_path),           # notebook de entrada
        str(out_path),          # notebook de saída (com outputs gravados)
        "--kernel", "python3",
        "--execution-timeout", "14400", 
        "--progress-bar",
    ]

    # Ambiente com PYTHONPATH corrigido para o processo filho
    env = _build_env()

    _log(f"  PYTHONPATH → {env['PYTHONPATH']}")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(ROOT),          # cwd = raiz do projeto
        capture_output=False,   # exibe stdout/stderr em tempo real
    )
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        _log(f"\n  [OK] Concluído em {_fmt_duration(elapsed)}")
        return True
    else:
        _log(f"\n  [FALHA] papermill encerrou com código {result.returncode}")
        _log(f"  Tempo até a falha: {_fmt_duration(elapsed)}")
        _log(f"  Notebook com traceback salvo em: {out_path}")
        return False


# ---------------------------------------------------------------------------
# Coleta de métricas via export_metrics_to_csv.py
# ---------------------------------------------------------------------------

def collect_metrics(dry_run: bool) -> None:
    """Chama export_metrics_to_csv via subprocess para manter o isolamento."""
    export_script = ROOT / "src" / "utils" / "export_metrics_to_csv.py"
    output_csv    = ROOT / "results" / "baseline_metrics.csv"

    _log(f"\n{'='*60}")
    _log("  COLETA DE MÉTRICAS")
    _log(f"  Script : {export_script.relative_to(ROOT)}")
    _log(f"  Saída  : {output_csv.relative_to(ROOT)}")
    _log(f"{'='*60}")

    if dry_run:
        _log("  [DRY-RUN] Pulando exportação real.")
        return

    result = subprocess.run(
        [sys.executable, str(export_script), "--detail"],
        cwd=str(ROOT),
        capture_output=False,   # exibe stdout/stderr diretamente no terminal
    )

    if result.returncode != 0:
        _log(f"\n  [AVISO] export_metrics_to_csv.py encerrou com código {result.returncode}.")
        _log("          Verifique se algum experimento falhou antes da coleta.")
    else:
        _log(f"\n  [OK] CSV gerado: {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Orquestrador da baseline: executa os notebooks na ordem correta "
            "e compila os resultados em CSV."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lista o pipeline sem executar nenhum notebook.",
    )
    parser.add_argument(
        "--from",
        dest="from_nb",
        metavar="NOTEBOOK_STEM",
        default=None,
        help=(
            "Retoma a execução a partir do notebook informado (stem do arquivo, "
            "ex.: arima_mlp). Útil para reiniciar após uma falha parcial."
        ),
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Não executa export_metrics_to_csv.py ao final.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if not args.dry_run:
        _check_papermill()

    start_idx = _resolve_start_index(args.from_nb)
    pipeline  = PIPELINE[start_idx:]

    _log("\n" + "="*60)
    _log("  BASELINE COMPLETA — ORQUESTRADOR")
    _log(f"  Séries : importadas de config.BASE_NAME_LIST")
    _log(f"  Etapas : {len(pipeline)} notebook(s) a executar")
    if args.dry_run:
        _log("  Modo   : DRY-RUN (sem execução real)")
    _log("="*60 + "\n")

    # Lista o pipeline completo antecipadamente
    for i, step in enumerate(pipeline):
        _log(f"  {i+1}. {step['label']}  →  {step['notebook'].name}")

    output_dir = OUTPUT_BASE
    total_start = time.perf_counter()
    failed: list[str] = []

    for step in pipeline:
        success = run_notebook(step, output_dir, dry_run=args.dry_run)
        if not success:
            failed.append(step["notebook"].stem)
            _log("\n[AVISO] Falha detectada. Continuando para o próximo notebook...")

    # Coleta de métricas
    if not args.skip_metrics:
        collect_metrics(dry_run=args.dry_run)

    # Relatório final
    total_elapsed = time.perf_counter() - total_start
    _log(f"\n{'='*60}")
    _log(f"  RESUMO FINAL")
    _log(f"  Tempo total : {_fmt_duration(total_elapsed)}")
    _log(f"  Sucesso     : {len(pipeline) - len(failed)}/{len(pipeline)} notebook(s)")

    if failed:
        _log(f"  Falhas      : {', '.join(failed)}")
        _log("  Use --from <notebook> para retomar a partir da falha.")
        sys.exit(1)
    else:
        _log("  Status      : TODOS OS EXPERIMENTOS CONCLUÍDOS COM SUCESSO")
        _log(f"  CSV gerado  : results/baseline_metrics.csv")
    _log("="*60 + "\n")


if __name__ == "__main__":
    main()
