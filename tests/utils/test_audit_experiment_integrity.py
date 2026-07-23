"""
Testes de src/utils/audit_experiment_integrity.py -- Tarefa 7.2 do roadmap:
auditoria unica, reutilizavel e permanente de conformidade metodologica.

As funcoes puras de checagem (check_*) sao testadas isoladamente com dados
sinteticos -- o mesmo padrao ja usado em test_compare_fs_vs_baseline.py.
Um teste de integracao real roda a auditoria completa contra os dados JA
existentes em data/result/ (sem gerar nada novo), com reconstruct_cv=False
para ficar rapido -- a evidencia real da reconstrucao de cv fica para a
execucao manual do script (relatorio final), nao para a suite de CI.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

import config
from model import generics
from model.feature_selection import TimeSeriesFeatureSelector
from utils import audit_experiment_integrity as audit


class _FakeExperiment:
    """Double de ResultExp/Additive/SKlearnModel para os testes de integracao
    de audit_fs_experiment. Definida no nivel do modulo (nao aninhada dentro
    de um metodo de teste) porque pickle nao serializa classes locais --
    generics.save_result usa pickle.dump de verdade."""

    def __init__(self, model):
        self.model = model
        self.experiment_params = {"diff_kpss": False, "horizon": 1}
        self.metrics_results = {"test_metrics": {"RMSE": 1.0}}


# ---------------------------------------------------------------------------
# check_pkl_count_and_naming
# ---------------------------------------------------------------------------

class TestCheckPklCountAndNaming:
    def test_pass_when_all_single_model_files_present(self, tmp_path):
        for series in ["airlines", "austres"]:
            (tmp_path / f"{series}_1mlpftest.pkl").touch()

        finding = audit.check_pkl_count_and_naming(tmp_path, ["airlines", "austres"], "mlpftest", is_hybrid=False)

        assert finding.status == "PASS"

    def test_pass_when_all_hybrid_files_present_including_copied_arima(self, tmp_path):
        for series in ["airlines", "austres"]:
            (tmp_path / f"{series}_1asftest.pkl").touch()
            (tmp_path / f"{series}_1arima.pkl").touch()

        finding = audit.check_pkl_count_and_naming(tmp_path, ["airlines", "austres"], "asftest", is_hybrid=True)

        assert finding.status == "PASS"

    def test_fail_when_missing_file(self, tmp_path):
        (tmp_path / "airlines_1mlpftest.pkl").touch()
        # austres ausente

        finding = audit.check_pkl_count_and_naming(tmp_path, ["airlines", "austres"], "mlpftest", is_hybrid=False)

        assert finding.status == "FAIL"
        assert "faltando" in finding.detail

    def test_fail_when_contaminated_by_another_family(self, tmp_path):
        for series in ["airlines", "austres"]:
            (tmp_path / f"{series}_1mlpftest.pkl").touch()
        (tmp_path / "airlines_1amv1ftest.pkl").touch()  # arquivo da familia hibrida, nao devia estar aqui

        finding = audit.check_pkl_count_and_naming(tmp_path, ["airlines", "austres"], "mlpftest", is_hybrid=False)

        assert finding.status == "FAIL"
        assert "contaminacao" in finding.detail

    def test_fail_when_directory_missing(self, tmp_path):
        finding = audit.check_pkl_count_and_naming(tmp_path / "nao_existe", ["airlines"], "mlpftest", is_hybrid=False)

        assert finding.status == "FAIL"


# ---------------------------------------------------------------------------
# check_hyperparameter_parity
# ---------------------------------------------------------------------------

class TestCheckHyperparameterParity:
    def test_pass_when_mlp_matches_baseline(self):
        estimator = MLPRegressor(activation="logistic", solver="lbfgs", hidden_layer_sizes=20, max_iter=1000)

        finding = audit.check_hyperparameter_parity(
            estimator,
            fixed_params={"activation": "logistic", "solver": "lbfgs"},
            grid_params={"hidden_layer_sizes": [10, 20, 50], "max_iter": [1000]},
        )

        assert finding.status == "PASS"

    def test_fail_when_fixed_param_diverges(self):
        """Achado real da Tarefa 3.8/3.9: exatamente este cenario (activation
        divergente do baseline) foi um bug de verdade -- a checagem precisa
        pegar isso."""
        estimator = MLPRegressor(activation="identity", solver="lbfgs", hidden_layer_sizes=20, max_iter=1000)

        finding = audit.check_hyperparameter_parity(
            estimator,
            fixed_params={"activation": "logistic", "solver": "lbfgs"},
            grid_params={"hidden_layer_sizes": [10, 20, 50], "max_iter": [1000]},
        )

        assert finding.status == "FAIL"
        assert "activation" in finding.detail

    def test_fail_when_grid_param_outside_baseline_grid(self):
        estimator = MLPRegressor(activation="logistic", solver="lbfgs", hidden_layer_sizes=999, max_iter=1000)

        finding = audit.check_hyperparameter_parity(
            estimator,
            fixed_params={"activation": "logistic", "solver": "lbfgs"},
            grid_params={"hidden_layer_sizes": [10, 20, 50], "max_iter": [1000]},
        )

        assert finding.status == "FAIL"
        assert "hidden_layer_sizes" in finding.detail

    def test_pass_when_svr_matches_baseline(self):
        estimator = SVR(C=100, gamma="auto", kernel="rbf", epsilon=0.01, tol=0.001)

        finding = audit.check_hyperparameter_parity(
            estimator,
            fixed_params={"gamma": "auto", "kernel": "rbf", "tol": 0.001},
            grid_params={"C": [10, 100, 1000], "epsilon": [0.1, 0.01, 0.001]},
        )

        assert finding.status == "PASS"


# ---------------------------------------------------------------------------
# check_experiment_params
# ---------------------------------------------------------------------------

class TestCheckExperimentParams:
    def test_pass_single_model(self):
        finding = audit.check_experiment_params(
            {"diff_kpss": False, "horizon": 1}, diff_kpss_expected=False, is_hybrid=False
        )
        assert finding.status == "PASS"

    def test_pass_hybrid_with_linear_model_name(self):
        finding = audit.check_experiment_params(
            {"diff_kpss": False, "horizon": 1, "linear_model_name": "1arima"},
            diff_kpss_expected=False,
            is_hybrid=True,
        )
        assert finding.status == "PASS"

    def test_fail_diff_kpss_mismatch(self):
        """Achado real da Tarefa 3.9/5.1: exatamente este cenario (diff_kpss
        desatualizado no baseline persistido) foi um bug de verdade."""
        finding = audit.check_experiment_params(
            {"diff_kpss": True, "horizon": 1}, diff_kpss_expected=False, is_hybrid=False
        )
        assert finding.status == "FAIL"
        assert "diff_kpss" in finding.detail

    def test_fail_hybrid_missing_linear_model_name(self):
        finding = audit.check_experiment_params(
            {"diff_kpss": False, "horizon": 1}, diff_kpss_expected=False, is_hybrid=True
        )
        assert finding.status == "FAIL"
        assert "linear_model_name" in finding.detail


# ---------------------------------------------------------------------------
# check_n_reps
# ---------------------------------------------------------------------------

class TestCheckNReps:
    def test_pass_mlp_based(self):
        assert audit.check_n_reps(10, 10).status == "PASS"

    def test_pass_svr_based(self):
        assert audit.check_n_reps(1, 1).status == "PASS"

    def test_fail_wrong_count(self):
        finding = audit.check_n_reps(10, 1)
        assert finding.status == "FAIL"


# ---------------------------------------------------------------------------
# check_cv_is_time_series_split
# ---------------------------------------------------------------------------

class TestCheckCvIsTimeSeriesSplit:
    def test_pass_time_series_split(self):
        assert audit.check_cv_is_time_series_split(TimeSeriesSplit).status == "PASS"

    def test_fail_kfold(self):
        """Regra nao-negociavel do projeto inteiro -- se algum dia isso
        aparecer de verdade, precisa ser FAIL, nao ATENCAO."""
        finding = audit.check_cv_is_time_series_split(KFold)
        assert finding.status == "FAIL"

    def test_atencao_when_reconstruction_failed(self):
        finding = audit.check_cv_is_time_series_split(None)
        assert finding.status == "ATENCAO"


# ---------------------------------------------------------------------------
# check_lag_size
# ---------------------------------------------------------------------------

class TestCheckLagSize:
    def test_pass_matches_reference(self):
        assert audit.check_lag_size(20, "airlines").status == "PASS"

    def test_atencao_diverges_from_reference(self):
        finding = audit.check_lag_size(15, "airlines")
        assert finding.status == "ATENCAO"

    def test_na_unknown_series(self):
        finding = audit.check_lag_size(5, "serie_desconhecida")
        assert finding.status == "N/A"


# ---------------------------------------------------------------------------
# check_baseline_hashes
# ---------------------------------------------------------------------------

class TestCheckBaselineHashes:
    def _write_pkl_and_reference(self, tmp_path, content=b"conteudo original"):
        baseline_dir = tmp_path / "chamados"
        baseline_dir.mkdir()
        pkl_path = baseline_dir / "airlines_1amv1.pkl"
        pkl_path.write_bytes(content)

        import hashlib
        reference_path = tmp_path / "reference.json"
        reference_path.write_text(
            json.dumps({"files": {"airlines_1amv1.pkl": hashlib.sha256(content).hexdigest()}}), encoding="utf-8"
        )
        return baseline_dir, reference_path

    def test_pass_when_hash_matches(self, tmp_path):
        baseline_dir, reference_path = self._write_pkl_and_reference(tmp_path)

        findings = audit.check_baseline_hashes(baseline_dir, reference_path)

        assert all(f.status == "PASS" for f in findings)

    def test_fail_when_file_was_modified(self, tmp_path):
        """Prova mecanica: se o baseline for reescrito (ex. regenerado por
        engano) sem atualizar a referencia, isso precisa ser FAIL, nao
        silenciosamente ignorado."""
        baseline_dir, reference_path = self._write_pkl_and_reference(tmp_path)
        (baseline_dir / "airlines_1amv1.pkl").write_bytes(b"conteudo MODIFICADO")

        findings = audit.check_baseline_hashes(baseline_dir, reference_path)

        assert any(f.status == "FAIL" for f in findings)

    def test_fail_when_referenced_file_disappeared(self, tmp_path):
        baseline_dir, reference_path = self._write_pkl_and_reference(tmp_path)
        (baseline_dir / "airlines_1amv1.pkl").unlink()

        findings = audit.check_baseline_hashes(baseline_dir, reference_path)

        assert any(f.status == "FAIL" for f in findings)

    def test_atencao_when_reference_file_missing(self, tmp_path):
        findings = audit.check_baseline_hashes(tmp_path, tmp_path / "nao_existe.json")

        assert len(findings) == 1
        assert findings[0].status == "ATENCAO"


# ---------------------------------------------------------------------------
# check_gamma_provisional_note
# ---------------------------------------------------------------------------

class TestCheckGammaProvisionalNote:
    def test_pass_when_note_present_and_complete(self, tmp_path):
        doc = tmp_path / "PLANO.md"
        doc.write_text(
            "### 1.11 NOTA PROVISORIA -- gamma\n"
            "gamma='auto' e calculado como 1 / n_features, entao muda com FS.\n",
            encoding="utf-8",
        )

        finding = audit.check_gamma_provisional_note(doc)

        assert finding.status == "PASS"

    def test_fail_when_note_absent(self, tmp_path):
        """Achado que a Tarefa 7.2 precisa detectar: nota provisoria removida/
        resolvida sem deixar registro."""
        doc = tmp_path / "PLANO.md"
        doc.write_text("# Documento sem nenhuma mencao a gamma\n", encoding="utf-8")

        finding = audit.check_gamma_provisional_note(doc)

        assert finding.status == "FAIL"


# ---------------------------------------------------------------------------
# Integracao -- audit_fs_experiment com .pkl sintetico real (generics.save_result)
# ---------------------------------------------------------------------------

class TestAuditFsExperimentIntegration:
    def _save_fs_pkl(self, experiment_id, base_name, model_name_suffix, strategy, n_reps):
        model = Pipeline([
            ("selector", TimeSeriesFeatureSelector(strategy=strategy, k=3, random_state=0)),
            ("estimator", MLPRegressor(activation="logistic", solver="lbfgs", hidden_layer_sizes=20, max_iter=1000)),
        ])
        rng = np.random.RandomState(0)
        X = rng.normal(size=(30, 6))
        y = 5.0 * X[:, 0] - 3.0 * X[:, 2] + 0.01 * rng.normal(size=30)
        model.fit(X, y)

        fold, title = generics.format_names(experiment_id, base_name, f"1{model_name_suffix}")
        entries = [{"experiment": _FakeExperiment(model), "val_metric": 1.0} for _ in range(n_reps)]
        generics.save_result(fold, title, entries)

    def test_detects_correct_experiment_as_all_pass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        experiment_id = "chamados_v4_fs_mlp_ftest"
        for series in ["airlines.txt", "austres.txt"]:
            self._save_fs_pkl(experiment_id, series, "mlpftest", "f_test", n_reps=10)

        family_spec = audit.FAMILY_SPECS["mlp"]
        findings = audit.audit_fs_experiment(
            family_spec, "f_test", "ftest", tmp_path, ["airlines", "austres"], reconstruct_cv=False
        )

        naming = [f for f in findings if f.check == "pkl_count_and_naming"]
        assert naming[0].status == "PASS"
        hp = [f for f in findings if f.check == "hyperparameter_parity"]
        assert all(f.status == "PASS" for f in hp)
        nreps = [f for f in findings if f.check == "n_reps"]
        assert all(f.status == "PASS" for f in nreps)

    def test_detects_wrong_n_reps_as_fail(self, tmp_path, monkeypatch):
        """Achado real de code-review possivel: uma familia MLP-based com
        model_exec errado (ex. 1 em vez de 10) precisa ser pego."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        experiment_id = "chamados_v4_fs_mlp_ftest"
        for series in ["airlines.txt", "austres.txt"]:
            self._save_fs_pkl(experiment_id, series, "mlpftest", "f_test", n_reps=1)  # deveria ser 10

        family_spec = audit.FAMILY_SPECS["mlp"]
        findings = audit.audit_fs_experiment(
            family_spec, "f_test", "ftest", tmp_path, ["airlines", "austres"], reconstruct_cv=False
        )

        nreps = [f for f in findings if f.check == "n_reps"]
        assert all(f.status == "FAIL" for f in nreps)


# ---------------------------------------------------------------------------
# Integracao real -- roda a auditoria completa contra os dados JA existentes
# em data/result/, sem gerar nada novo. reconstruct_cv=False para ficar
# rapido (a reconstrucao de verdade e evidencia real fica para a execucao
# manual do script, que produz o relatorio final).
# ---------------------------------------------------------------------------

class TestRunAuditAgainstRealData:
    def test_completes_without_crashing_and_returns_findings(self):
        findings = audit.run_audit(reconstruct_cv=False)

        assert len(findings) > 0
        # todo finding tem um status valido
        assert all(f.status in ("PASS", "FAIL", "ATENCAO", "N/A") for f in findings)

    def test_baseline_hashes_all_pass_against_the_reference_just_generated(self):
        """A referencia (data/result/chamados_baseline_reference_hashes.json)
        foi gerada nesta mesma tarefa a partir do estado atual, ja confirmado
        correto pelos portoes anteriores -- deve bater 100% agora."""
        findings = audit.check_baseline_hashes(audit.DEFAULT_BASELINE_DIR, audit.DEFAULT_BASELINE_HASH_REFERENCE)

        fails = [f for f in findings if f.status == "FAIL"]
        assert fails == [], f"Baselines deveriam bater com a referencia recem-gerada: {fails}"

    def test_gamma_provisional_note_is_present(self):
        finding = audit.check_gamma_provisional_note(audit.DEFAULT_PLANO_ARQUITETURA)
        assert finding.status == "PASS"
