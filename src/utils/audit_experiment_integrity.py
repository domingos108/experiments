"""
audit_experiment_integrity.py
------------------------------
Auditoria unica, reutilizavel e permanente de conformidade metodologica sobre
toda a arvore de data/result/ relevante -- os 5 baselines protegidos
(CLAUDE.md Secao 3) + os experimentos de Feature Selection (familia x
metodo). Consolida, de forma automatizada e reexecutavel, a mesma classe de
checagem que os portoes de validacao manuais (Tarefas 5-gate/6-gate/7-gate)
ja faziam ad-hoc para cada familia.

Nao corrige nada -- so diagnostica e reporta. Qualquer divergencia real fica
para o pesquisador decidir.

Uso
---
    python src/utils/audit_experiment_integrity.py \
        --output results/integrity_audit_report_v1.md

    # Pulando a checagem cara de reconstrucao do cv (Lasso/RFECV):
    python src/utils/audit_experiment_integrity.py --skip-cv-reconstruction
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
from model import generics
from model.grid_search_exp import resolve_lag_size
from utils.export_metrics_to_csv import _load_pkl, _unwrap_entry

DEFAULT_OUTPUT = ROOT / "results" / "integrity_audit_report_v1.md"
DEFAULT_BASELINE_DIR = Path(config.MODEL_DATA_PATH) / "chamados"
DEFAULT_BASELINE_HASH_REFERENCE = Path(config.MODEL_DATA_PATH) / "chamados_baseline_reference_hashes.json"
DEFAULT_PLANO_ARQUITETURA = ROOT / "PLANO_ARQUITETURA.md"

# 4 series usadas em todos os experimentos de FS ate agora (Tarefa 1,
# PLANO_ARQUITETURA.md Secao 3, item 1) -- duplicado aqui (nao importado de
# tests/) porque src/ nao deve depender de tests/; cada notebook de FS ja
# redefine essa mesma lista localmente, mesmo padrao.
FS_DEV_SERIES = ["airlines", "austres", "coloradoRiver", "sunspot"]

# lag_size='auto' resolve para o MESMO valor nas 5 familias (confirmado com
# dado real nas Tarefas 5/6/7 -- a serie/residuo em si nao depende do
# estimador a jusante). Referencia unica; se alguma familia divergir disso,
# a auditoria sinaliza.
EXPECTED_LAG_SIZE = {
    "airlines": 20,
    "austres": 1,
    "coloradoRiver": 16,
    "sunspot": 9,
}

METHODS = [
    ("f_test", "ftest"),
    ("mutual_info", "mutualinfo"),
    ("rf_embedded", "rfembedded"),
    ("lasso", "lasso"),
    ("rfecv", "rfecv"),
]
METHODS_WITH_INTERNAL_CV = {"lasso", "rfecv"}
METHODS_WITH_SELECTOR_K_GRID = {"f_test", "mutual_info"}

# Serie preferida para a reconstrucao (cara) do cv de Lasso/RFECV -- feita 1x
# por (familia, metodo), nao 1x por serie, para manter o audit rapido.
# 'austres' e explicitamente evitada aqui: N_Features_Total=1 aciona a
# guarda de 1-feature do RFECV (Tarefa 4), que NUNCA chega a construir
# RFECV(...) -- reconstruir com ela daria um falso "nao foi possivel
# completar" em vez de uma prova real do cv.
PREFERRED_CV_RECONSTRUCTION_SERIES = "airlines"


@dataclass
class FamilySpec:
    """Descreve uma familia de experimentos (baseline + convencao de FS).
    Nova familia = uma nova entrada aqui, sem tocar no resto do script."""

    name: str
    baseline_model_name: str  # sufixo do baseline, ex. 'amv1', 'mlp' -- SEM o prefixo de horizon
    is_hybrid: bool  # True = Additive (copia '1arima' pra dentro de cada pasta de FS)
    supports_fs: bool  # False para 'arima' (sem variante de FS -- so referencia linear)
    estimator_step_name: str  # nome do step do Pipeline ('estimator') -- constante hoje, exposto para o futuro
    fixed_params: dict = field(default_factory=dict)  # hiperparametros FIXOS (nao fazem parte do grid) -- match exato
    grid_params: dict = field(default_factory=dict)  # hiperparametros do GRID -- checa pertencimento, nao igualdade
    model_exec_expected: int | None = None
    diff_kpss_expected: bool | None = None
    fs_experiment_id_template: str = ""  # ex. 'chamados_v4_fs_mlp_{method_slug}'
    fs_model_name_template: str = ""  # ex. 'mlp{method_slug}'


FAMILY_SPECS: dict[str, FamilySpec] = {
    "arima": FamilySpec(
        name="arima",
        baseline_model_name="arima",
        is_hybrid=False,
        supports_fs=False,
        estimator_step_name="",
    ),
    "mlp": FamilySpec(
        name="mlp",
        baseline_model_name="mlp",
        is_hybrid=False,
        supports_fs=True,
        estimator_step_name="estimator",
        fixed_params={"activation": "logistic", "solver": "lbfgs"},
        grid_params={"hidden_layer_sizes": [10, 20, 50], "max_iter": [1000]},
        model_exec_expected=10,
        diff_kpss_expected=False,
        fs_experiment_id_template="chamados_v4_fs_mlp_{method_slug}",
        fs_model_name_template="mlp{method_slug}",
    ),
    "svr": FamilySpec(
        name="svr",
        baseline_model_name="svr",
        is_hybrid=False,
        supports_fs=True,
        estimator_step_name="estimator",
        fixed_params={"gamma": "auto", "kernel": "rbf", "tol": 0.001},
        grid_params={"C": [10, 100, 1000], "epsilon": [0.1, 0.01, 0.001]},
        model_exec_expected=1,
        diff_kpss_expected=True,
        fs_experiment_id_template="chamados_v4_fs_svr_{method_slug}",
        fs_model_name_template="svr{method_slug}",
    ),
    "arima_mlp": FamilySpec(
        name="arima_mlp",
        baseline_model_name="amv1",
        is_hybrid=True,
        supports_fs=True,
        estimator_step_name="estimator",
        fixed_params={"activation": "logistic", "solver": "lbfgs"},
        grid_params={"hidden_layer_sizes": [10, 20, 50], "max_iter": [1000]},
        model_exec_expected=10,
        diff_kpss_expected=False,
        fs_experiment_id_template="chamados_v4_fs_{method_slug}",
        fs_model_name_template="amv1{method_slug}",
    ),
    "arima_svr": FamilySpec(
        name="arima_svr",
        baseline_model_name="as",
        is_hybrid=True,
        supports_fs=True,
        estimator_step_name="estimator",
        fixed_params={"gamma": "auto", "kernel": "rbf", "tol": 0.001},
        grid_params={"C": [10, 100, 1000], "epsilon": [0.1, 0.01, 0.001]},
        model_exec_expected=1,
        diff_kpss_expected=False,
        fs_experiment_id_template="chamados_v4_fs_arimasvr_{method_slug}",
        fs_model_name_template="as{method_slug}",
    ),
}


@dataclass
class Finding:
    experiment: str  # ex. "mlp/ftest", "baseline/1amv1"
    check: str
    status: str  # "PASS" | "FAIL" | "ATENCAO" | "N/A"
    detail: str = ""


# ---------------------------------------------------------------------------
# Checagem 1 -- contagem e nomenclatura de .pkl
# ---------------------------------------------------------------------------

def check_pkl_count_and_naming(
    experiment_dir: Path, series_list: list[str], model_name_suffix: str, is_hybrid: bool
) -> Finding:
    """`model_name_suffix` e o sufixo SEM o prefixo de horizon (ex. 'mlpftest'),
    que GridSearch prefixa com '1' automaticamente -- checa '1<sufixo>.pkl'."""
    if not experiment_dir.exists():
        return Finding("", "pkl_count_and_naming", "FAIL", f"Diretorio nao existe: {experiment_dir}")

    expected = {f"{series}_1{model_name_suffix}.pkl" for series in series_list}
    if is_hybrid:
        expected |= {f"{series}_1arima.pkl" for series in series_list}

    actual = {p.name for p in experiment_dir.glob("*.pkl")}

    missing = expected - actual
    unexpected = actual - expected

    if missing or unexpected:
        detail_parts = []
        if missing:
            detail_parts.append(f"faltando: {sorted(missing)}")
        if unexpected:
            detail_parts.append(f"arquivo(s) INESPERADO(S)/contaminacao: {sorted(unexpected)}")
        return Finding("", "pkl_count_and_naming", "FAIL", "; ".join(detail_parts))

    return Finding("", "pkl_count_and_naming", "PASS", f"{len(expected)} arquivo(s) conferem")


# ---------------------------------------------------------------------------
# Checagem 2 -- paridade de hiperparametros com o baseline da familia
# ---------------------------------------------------------------------------

def check_hyperparameter_parity(
    estimator, fixed_params: dict, grid_params: dict
) -> Finding:
    """`estimator` e o objeto ja fitted (MLPRegressor/SVR) extraido do
    Pipeline. fixed_params exige igualdade exata; grid_params exige que o
    valor usado pertenca a lista do grid do baseline."""
    problems = []
    for key, expected_value in fixed_params.items():
        actual_value = getattr(estimator, key, "<ausente>")
        if actual_value != expected_value:
            problems.append(f"{key}={actual_value!r} (esperado {expected_value!r})")

    for key, allowed_values in grid_params.items():
        actual_value = getattr(estimator, key, "<ausente>")
        if actual_value not in allowed_values:
            problems.append(f"{key}={actual_value!r} (fora do grid do baseline {allowed_values!r})")

    if problems:
        return Finding("", "hyperparameter_parity", "FAIL", "; ".join(problems))
    return Finding("", "hyperparameter_parity", "PASS", "todos os hiperparametros conferem com o baseline")


# ---------------------------------------------------------------------------
# Checagem 3 -- consistencia de experiment_params
# ---------------------------------------------------------------------------

def check_experiment_params(
    experiment_params: dict, diff_kpss_expected: bool, is_hybrid: bool, horizon_expected: int = 1
) -> Finding:
    problems = []
    actual_diff_kpss = experiment_params.get("diff_kpss")
    if actual_diff_kpss != diff_kpss_expected:
        problems.append(f"diff_kpss={actual_diff_kpss!r} (esperado {diff_kpss_expected!r})")

    actual_horizon = experiment_params.get("horizon")
    if actual_horizon != horizon_expected:
        problems.append(f"horizon={actual_horizon!r} (esperado {horizon_expected!r})")

    if is_hybrid:
        actual_linear = experiment_params.get("linear_model_name")
        if actual_linear != "1arima":
            problems.append(f"linear_model_name={actual_linear!r} (esperado '1arima')")

    if problems:
        return Finding("", "experiment_params", "FAIL", "; ".join(problems))
    return Finding("", "experiment_params", "PASS", "diff_kpss/horizon/linear_model_name conferem")


# ---------------------------------------------------------------------------
# Checagem 4 -- n_reps / model_exec
# ---------------------------------------------------------------------------

def check_n_reps(n_reps: int, model_exec_expected: int) -> Finding:
    if n_reps != model_exec_expected:
        return Finding(
            "", "n_reps", "FAIL",
            f"{n_reps} repeticao(oes) persistida(s), esperado {model_exec_expected} "
            f"(estocastico=10 para MLP-based, deterministico=1 para SVR-based)",
        )
    return Finding("", "n_reps", "PASS", f"{n_reps} repeticao(oes), conforme esperado")


# ---------------------------------------------------------------------------
# Checagem 5 -- cv do Lasso/RFECV (TimeSeriesSplit, nunca KFold)
# ---------------------------------------------------------------------------

def _reconstruct_and_capture_cv(
    family_spec: FamilySpec, method: str, series: str, experiment_id: str,
    persisted_k, persisted_random_state, experiment_params: dict,
) -> tuple[type | None, str | None]:
    """Reconstroi o X_train/y_train real (mesma tecnica ja usada nas Tarefas
    3.6/4/8) e refaz SO o fit do seletor (com um DummyRegressor no lugar do
    estimador de producao), capturando o TIPO do objeto `cv` passado para
    LassoCV/RFECV via monkeypatch -- prova mecanica, nao leitura do
    codigo-fonte, de que o cv usado de fato foi TimeSeriesSplit. Retorna
    (classe_capturada, mensagem_de_erro) -- exatamente um dos dois e None.

    O espiao PRECISA ser uma funcao que devolve uma instancia real de
    LassoCV/RFECV (nao uma subclasse com `__init__(self, *args, **kwargs)`):
    SelectFromModel.fit() chama clone(self.estimator) internamente, e o
    clone() do sklearn exige que _get_param_names() funcione via inspecao da
    assinatura do __init__ -- uma subclasse com *args/**kwargs quebra isso
    com RuntimeError (achado real, Tarefa 7.2: a primeira versao deste
    espiao para Lasso usava subclasse e falhava silenciosamente em todo
    experimento com lasso, mascarado por um `except Exception: return None`
    que engolia o erro real -- corrigido em ambos os pontos)."""
    import model.feature_selection as fs_module
    from model import generics as generics_module
    from model.feature_selection import TimeSeriesFeatureSelector
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline

    captured = {}
    real_lasso_cv = fs_module.LassoCV
    real_rfecv = fs_module.RFECV

    def _spy_lasso_cv(*args, **kwargs):
        captured["cv_type"] = type(kwargs.get("cv"))
        return real_lasso_cv(*args, **kwargs)

    def _spy_rfecv(*args, **kwargs):
        captured["cv_type"] = type(kwargs.get("cv"))
        return real_rfecv(*args, **kwargs)

    fs_module.LassoCV = _spy_lasso_cv
    fs_module.RFECV = _spy_rfecv
    try:
        probe = Pipeline([
            ("selector", TimeSeriesFeatureSelector(strategy=method, k=persisted_k or 5, random_state=persisted_random_state)),
            ("estimator", DummyRegressor()),
        ])

        if family_spec.is_hybrid:
            from model.hybrid_system_exp import input_linear_info

            error_series, _ts_forecast, base_info, exec_config = input_linear_info(
                experiment_id, f"{series}.txt", experiment_params
            )
            generics_module.fit_predict_model(
                probe, pd.Series(error_series), True, base_info.lag_size_formated, exec_config, False,
            )
        else:
            from input.input import open_format_train_val_test

            exec_config = {
                "test_size": config.TEST_SIZE,
                "val_size": config.VAL_SIZE,
                "horizon": experiment_params.get("horizon", 1),
                "lag_size": "auto",
                "diff_kpss": experiment_params.get("diff_kpss", False),
                "normalize": True,
                "type_filter": experiment_params.get("type_filter", None),
            }
            base_info = open_format_train_val_test(f"{series}.txt", exec_config)
            x_train = base_info.df_train.drop(columns=["actual"]).values
            y_train = base_info.df_train["actual"].values
            probe.fit(x_train, y_train)
    except Exception as exc:  # reconstrucao pode falhar por motivo alheio ao cv em si -- reportar, nao engolir
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        fs_module.LassoCV = real_lasso_cv
        fs_module.RFECV = real_rfecv

    return captured.get("cv_type"), None


def check_cv_is_time_series_split(cv_type: type | None, error_message: str | None = None) -> Finding:
    from sklearn.model_selection import KFold, TimeSeriesSplit

    if cv_type is None:
        detail = f"reconstrucao nao pode ser completada: {error_message}" if error_message else "reconstrucao nao pode ser completada (motivo desconhecido)"
        return Finding("", "cv_time_series_split", "ATENCAO", detail)
    if cv_type is KFold:
        return Finding("", "cv_time_series_split", "FAIL", "cv=KFold (aleatorio!) -- vazamento temporal real")
    if not issubclass(cv_type, TimeSeriesSplit):
        return Finding("", "cv_time_series_split", "FAIL", f"cv={cv_type.__name__}, esperado TimeSeriesSplit")
    return Finding("", "cv_time_series_split", "PASS", "cv=TimeSeriesSplit confirmado por reconstrucao real")


# ---------------------------------------------------------------------------
# Checagem 6 -- lag_size='auto' real
# ---------------------------------------------------------------------------

def check_lag_size(actual_lag_size: int, series: str) -> Finding:
    expected = EXPECTED_LAG_SIZE.get(series)
    if expected is None:
        return Finding("", "lag_size", "N/A", f"serie {series!r} sem valor de referencia conhecido")
    if actual_lag_size != expected:
        return Finding(
            "", "lag_size", "ATENCAO",
            f"lag_size={actual_lag_size}, esperado {expected} (divergencia NAO documentada no RUNBOOK.md)",
        )
    return Finding("", "lag_size", "PASS", f"lag_size={actual_lag_size}, conforme RUNBOOK.md")


# ---------------------------------------------------------------------------
# Checagem 7 -- hashes dos 5 baselines protegidos
# ---------------------------------------------------------------------------

def check_baseline_hashes(baseline_dir: Path, reference_path: Path) -> list[Finding]:
    import hashlib

    if not reference_path.exists():
        return [Finding(
            "baseline/*", "baseline_hash", "ATENCAO",
            f"arquivo de referencia nao encontrado ({reference_path}) -- rode com "
            "--write-baseline-reference para criar um novo a partir do estado atual.",
        )]

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_hashes = reference["files"]

    findings = []
    actual_files = {p.name: p for p in baseline_dir.glob("*.pkl")}

    for name, expected_hash in reference_hashes.items():
        pkl_path = actual_files.get(name)
        if pkl_path is None:
            findings.append(Finding(f"baseline/{name}", "baseline_hash", "FAIL", "arquivo referenciado sumiu"))
            continue
        actual_hash = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            findings.append(Finding(
                f"baseline/{name}", "baseline_hash", "FAIL",
                "hash DIVERGENTE do valor de referencia -- baseline foi modificado sem atualizar a referencia",
            ))
        else:
            findings.append(Finding(f"baseline/{name}", "baseline_hash", "PASS", "hash confere com a referencia"))

    new_files = set(actual_files) - set(reference_hashes)
    for name in sorted(new_files):
        findings.append(Finding(f"baseline/{name}", "baseline_hash", "ATENCAO", "arquivo novo, fora da referencia"))

    return findings


# ---------------------------------------------------------------------------
# Checagem 8 -- nota provisoria de gamma='auto' presente
# ---------------------------------------------------------------------------

def check_gamma_provisional_note(plano_arquitetura_path: Path) -> Finding:
    if not plano_arquitetura_path.exists():
        return Finding("gamma_note", "gamma_provisional_note", "ATENCAO", f"{plano_arquitetura_path} nao encontrado")

    text = plano_arquitetura_path.read_text(encoding="utf-8")
    has_provisional_marker = "PROVIS" in text.upper() and "gamma" in text
    has_gamma_explanation = "1 / n_features" in text or "1/n_features" in text

    if has_provisional_marker and has_gamma_explanation:
        return Finding("gamma_note", "gamma_provisional_note", "PASS", "nota provisoria presente e completa")
    if has_provisional_marker or has_gamma_explanation:
        return Finding(
            "gamma_note", "gamma_provisional_note", "ATENCAO",
            "nota encontrada mas incompleta (falta o marcador PROVISORIO ou a explicacao de 1/n_features)",
        )
    return Finding(
        "gamma_note", "gamma_provisional_note", "FAIL",
        "nota provisoria sobre gamma='auto' NAO encontrada -- pode ter sido removida/resolvida sem registro",
    )


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------

def _read_fs_pkl_entries(experiment_dir: Path, series: str, model_name_suffix: str) -> list | None:
    pkl_path = experiment_dir / f"{series}_1{model_name_suffix}.pkl"
    if not pkl_path.exists():
        return None
    return _load_pkl(pkl_path)


def audit_fs_experiment(
    family_spec: FamilySpec, method: str, method_slug: str, base_dir: Path, series_list: list[str],
    reconstruct_cv: bool = True,
) -> list[Finding]:
    experiment_id = family_spec.fs_experiment_id_template.format(method_slug=method_slug)
    model_name_suffix = family_spec.fs_model_name_template.format(method_slug=method_slug)
    experiment_dir = base_dir / experiment_id
    experiment_label = f"{family_spec.name}/{method_slug}"

    findings: list[Finding] = []

    naming_finding = check_pkl_count_and_naming(experiment_dir, series_list, model_name_suffix, family_spec.is_hybrid)
    naming_finding.experiment = experiment_label
    findings.append(naming_finding)
    if naming_finding.status == "FAIL":
        return findings  # sem os .pkl certos, as checagens seguintes nao tem o que ler

    # 1 reconstrucao de cv por (familia, metodo), usando a serie preferida se
    # disponivel (senao a primeira da lista) -- nao 1x por serie, ver
    # PREFERRED_CV_RECONSTRUCTION_SERIES.
    cv_reconstruction_series = (
        PREFERRED_CV_RECONSTRUCTION_SERIES if PREFERRED_CV_RECONSTRUCTION_SERIES in series_list else series_list[0]
    )

    for series in series_list:
        entries = _read_fs_pkl_entries(experiment_dir, series, model_name_suffix)
        if not entries:
            findings.append(Finding(f"{experiment_label}/{series}", "pkl_readable", "FAIL", "nao encontrado/vazio"))
            continue

        result_exp, _val_metric = _unwrap_entry(entries[0])
        model = getattr(result_exp, "model", None)
        experiment_params = getattr(result_exp, "experiment_params", {})

        if model is not None and hasattr(model, "named_steps"):
            estimator = model.named_steps.get(family_spec.estimator_step_name)
            hp_finding = check_hyperparameter_parity(estimator, family_spec.fixed_params, family_spec.grid_params)
            hp_finding.experiment = f"{experiment_label}/{series}"
            findings.append(hp_finding)

            selector = model.named_steps.get("selector")
            if selector is not None and hasattr(selector, "n_features_in_"):
                lag_finding = check_lag_size(selector.n_features_in_, series)
                lag_finding.experiment = f"{experiment_label}/{series}"
                findings.append(lag_finding)

        ep_finding = check_experiment_params(experiment_params, family_spec.diff_kpss_expected, family_spec.is_hybrid)
        ep_finding.experiment = f"{experiment_label}/{series}"
        findings.append(ep_finding)

        nreps_finding = check_n_reps(len(entries), family_spec.model_exec_expected)
        nreps_finding.experiment = f"{experiment_label}/{series}"
        findings.append(nreps_finding)

        if method in METHODS_WITH_INTERNAL_CV and reconstruct_cv and series == cv_reconstruction_series:
            selector = model.named_steps.get("selector") if model is not None else None
            k = getattr(selector, "k", 5) if selector is not None else 5
            random_state = getattr(selector, "random_state", None) if selector is not None else None
            cv_type, error_message = _reconstruct_and_capture_cv(
                family_spec, method, series, experiment_id, k, random_state, experiment_params
            )
            cv_finding = check_cv_is_time_series_split(cv_type, error_message)
            cv_finding.experiment = f"{experiment_label} (amostra: {series})"
            findings.append(cv_finding)

    return findings


def audit_baseline(family_spec: FamilySpec, baseline_dir: Path) -> list[Finding]:
    """Baselines nao tem 'metodo' -- so confirma que o .pkl existe e (quando
    aplicavel) que o proprio baseline usa os hiperparametros fixos
    documentados. 'arima' e estruturalmente diferente (ResultExp sem
    .model/.experiment_params, ordem escolhida por auto_arima) -- N/A para
    as checagens que nao se aplicam a ele, nao FAIL."""
    findings = []
    label = f"baseline/1{family_spec.baseline_model_name}"

    for series in FS_DEV_SERIES:
        pkl_path = baseline_dir / f"{series}_1{family_spec.baseline_model_name}.pkl"
        if not pkl_path.exists():
            findings.append(Finding(f"{label}/{series}", "baseline_pkl_exists", "FAIL", "arquivo nao encontrado"))
            continue
        findings.append(Finding(f"{label}/{series}", "baseline_pkl_exists", "PASS", "arquivo presente"))

        entries = _load_pkl(pkl_path)
        result_exp, _ = _unwrap_entry(entries[0])
        model = getattr(result_exp, "model", None)
        experiment_params = getattr(result_exp, "experiment_params", None)

        if experiment_params is None:
            findings.append(Finding(f"{label}/{series}", "experiment_params", "N/A", "familia sem experiment_params (ex. ARIMA/auto_arima)"))
        else:
            ep_finding = check_experiment_params(experiment_params, family_spec.diff_kpss_expected, family_spec.is_hybrid)
            ep_finding.experiment = f"{label}/{series}"
            findings.append(ep_finding)

        if model is None or not hasattr(model, "get_params"):
            findings.append(Finding(f"{label}/{series}", "hyperparameter_parity", "N/A", "sem estimador sklearn expondo hiperparametros"))
        else:
            problems = [
                f"{k}={getattr(model, k, '<ausente>')!r} (esperado {v!r})"
                for k, v in family_spec.fixed_params.items()
                if getattr(model, k, "<ausente>") != v
            ]
            if problems:
                findings.append(Finding(f"{label}/{series}", "hyperparameter_parity", "FAIL", "; ".join(problems)))
            else:
                findings.append(Finding(f"{label}/{series}", "hyperparameter_parity", "PASS", "hiperparametros fixos conferem"))

    return findings


def run_audit(
    families: dict[str, FamilySpec] = FAMILY_SPECS,
    methods: list[tuple[str, str]] = METHODS,
    series_list: list[str] = FS_DEV_SERIES,
    base_dir: Path = Path(config.MODEL_DATA_PATH),
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    baseline_hash_reference: Path = DEFAULT_BASELINE_HASH_REFERENCE,
    plano_arquitetura_path: Path = DEFAULT_PLANO_ARQUITETURA,
    reconstruct_cv: bool = True,
) -> list[Finding]:
    all_findings: list[Finding] = []

    for family_spec in families.values():
        all_findings.extend(audit_baseline(family_spec, baseline_dir))

        if not family_spec.supports_fs:
            continue
        for method, method_slug in methods:
            all_findings.extend(
                audit_fs_experiment(family_spec, method, method_slug, base_dir, series_list, reconstruct_cv)
            )

    all_findings.extend(check_baseline_hashes(baseline_dir, baseline_hash_reference))
    all_findings.append(check_gamma_provisional_note(plano_arquitetura_path))

    return all_findings


# ---------------------------------------------------------------------------
# Relatorio Markdown
# ---------------------------------------------------------------------------

def render_markdown_report(findings: list[Finding]) -> str:
    n_pass = sum(1 for f in findings if f.status == "PASS")
    n_fail = sum(1 for f in findings if f.status == "FAIL")
    n_atencao = sum(1 for f in findings if f.status == "ATENCAO")
    n_na = sum(1 for f in findings if f.status == "N/A")
    total = len(findings)

    lines = [
        "# Auditoria de Integridade Metodologica -- v1",
        "",
        f"**Total de checagens: {total}** -- ✅ {n_pass} PASS, ❌ {n_fail} FAIL, ⚠️ {n_atencao} ATENÇÃO, ➖ {n_na} N/A",
        "",
    ]

    if n_fail == 0 and n_atencao == 0:
        lines.append("**Nenhuma divergência encontrada.**")
    else:
        lines.append("## Itens que exigem atenção")
        lines.append("")
        lines.append("| Experimento | Checagem | Status | Detalhe |")
        lines.append("|---|---|---|---|")
        for f in findings:
            if f.status in ("FAIL", "ATENCAO"):
                marker = "❌" if f.status == "FAIL" else "⚠️"
                lines.append(f"| {f.experiment} | {f.check} | {marker} {f.status} | {f.detail} |")
    lines.append("")

    lines.append("## Todas as checagens")
    lines.append("")
    lines.append("| Experimento | Checagem | Status | Detalhe |")
    lines.append("|---|---|---|---|")
    for f in findings:
        marker = {"PASS": "✅", "FAIL": "❌", "ATENCAO": "⚠️", "N/A": "➖"}[f.status]
        lines.append(f"| {f.experiment} | {f.check} | {marker} {f.status} | {f.detail} |")
    lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditoria de integridade metodologica dos experimentos de FS.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-cv-reconstruction", action="store_true", help="Pula a reconstrucao (mais lenta) do cv de Lasso/RFECV.")
    parser.add_argument(
        "--write-baseline-reference", action="store_true",
        help="Gera/sobrescreve o arquivo de referencia de hash dos baselines a partir do estado atual (use com cuidado -- so apos confirmar que os baselines estao corretos).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_baseline_reference:
        import hashlib
        from datetime import datetime

        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEFAULT_BASELINE_DIR.glob("*.pkl"))}
        out = {"generated_at": datetime.now().isoformat(), "files": files}
        DEFAULT_BASELINE_HASH_REFERENCE.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Referencia de hash escrita em {DEFAULT_BASELINE_HASH_REFERENCE} ({len(files)} arquivos).")
        return

    findings = run_audit(reconstruct_cv=not args.skip_cv_reconstruction)
    report = render_markdown_report(findings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    n_pass = sum(1 for f in findings if f.status == "PASS")
    n_fail = sum(1 for f in findings if f.status == "FAIL")
    n_atencao = sum(1 for f in findings if f.status == "ATENCAO")
    print(f"Relatorio gerado em {args.output}: {n_pass} PASS, {n_fail} FAIL, {n_atencao} ATENCAO.")


if __name__ == "__main__":
    main()
