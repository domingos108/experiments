"""
Teste de regressao (Tarefa 3.4, exigido pelo DoD): os .pkl JA EXISTENTES dos
5 baselines (formato antigo, gerados ANTES da chave `grid_search_history`
existir) continuam sendo lidos sem erro pelos consumidores atuais --
`metrics.open_fold_result` e `export_metrics_to_csv.collect_all_rows`.

Nao re-executa nenhum GridSearch -- so LE os .pkl reais ja commitados em
data/result/chamados/, provando compatibilidade retroativa (a chave nova e
estritamente aditiva, entao .pkl antigos sem ela continuam validos).
"""

from pathlib import Path

import config
from metrics.metrics import open_fold_result
from utils.export_metrics_to_csv import collect_all_rows


class TestExistingBaselinePklStillReadableAfterGridSearchHistoryChange:
    def test_metrics_open_fold_result_reads_real_baseline_pkl_without_error(self):
        df_mean, df_all, df_prevs = open_fold_result("chamados", "val_metrics", "RMSE")

        assert not df_mean.empty
        assert ("airlines", "1mlp") in df_mean.index or "1mlp" in df_mean.reset_index()["model"].values

    def test_export_metrics_to_csv_reads_real_baseline_pkl_without_error(self):
        result_dir = Path(config.MODEL_DATA_PATH) / "chamados"

        df_detail = collect_all_rows(result_dir)

        assert not df_detail.empty
        assert (df_detail["Modelo"] == "1mlp").any()
        assert (df_detail["Modelo"] == "1amv1").any()
