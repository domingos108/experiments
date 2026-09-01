# Tabela Consolidada — ARIMA, MLP, SVR (referência) x ARIMA-MLP, ARIMA-SVR (com FS)

Solicitada pelo orientador. Escopo: `ARIMA`, `MLP`, `SVR` aparecem apenas como referência (sem Feature Selection, por decisão do orientador — FS não é aplicado a modelos single isolados). `ARIMA-MLP` e `ARIMA-SVR` (via Additive) trazem o baseline sem FS e as 5 variantes de FS cada. Melhor RMSE de cada série em **negrito**. Fonte: `results/benchmark_master_with_sources_v1.csv` (Tarefa 9), validado pela Tarefa 13 (nenhum bug de configuração encontrado nos casos de empate).

> **Nota metodológica**: em `f_test`/`mutual_info`, quando o `k` vencedor do grid é maior ou igual ao número total de lags candidatos da série, `SelectKBest` se torna uma transformação identidade (usa todas as features) — o resultado idêntico ao baseline nesses casos é uma consequência matemática da definição de top-k, não uma falha do seletor. Confirmado com histórico completo de validação por `k` (Tarefa 13).


## airlines

| Família | Método | RMSE | MAPE | NFeatures | Fonte |
|---|---|---|---|---|---|
| ARIMA | sem Feature Selection | 19.7280 | 3.3153 | — | `results/baseline_metrics.csv` |
| MLP | sem Feature Selection | 27.6977 | 5.1091 | 20.0 | `results/baseline_metrics.csv` |
| SVR | sem Feature Selection | 19.3867 | 3.1112 | 20.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | sem Feature Selection | 17.4362 | 3.2519 | 20.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | f_test | 19.1139 | 3.2203 | 5.0 | `results/chamados_v4_fs_ftest/metrics.csv` |
| ARIMA-MLP | mutual_info | 19.6141 | 3.4503 | 9.0 | `results/chamados_v4_fs_mutualinfo/metrics.csv` |
| ARIMA-MLP | rf_embedded | 19.4301 | 3.2443 | 7.4 | `results/chamados_v4_fs_rfembedded/metrics.csv` |
| ARIMA-MLP | lasso | 19.7328 | 3.2817 | 1.0 | `results/chamados_v4_fs_lasso/metrics.csv` |
| ARIMA-MLP | rfecv | 18.0167 | 3.2756 | 16.7 | `results/chamados_v4_fs_rfecv/metrics.csv` |
| ARIMA-SVR | sem Feature Selection | **15.1665** | 3.0674 | 20.0 | `results/baseline_metrics.csv` |
| ARIMA-SVR | f_test | **15.1665** | 3.0674 | 20.0 | `results/chamados_v4_fs_arimasvr_ftest/metrics.csv` |
| ARIMA-SVR | mutual_info | **15.1665** | 3.0674 | 20.0 | `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv` |
| ARIMA-SVR | rf_embedded | 20.0391 | 3.5017 | 9.0 | `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv` |
| ARIMA-SVR | lasso | 19.2861 | 3.1099 | 1.0 | `results/chamados_v4_fs_arimasvr_lasso/metrics.csv` |
| ARIMA-SVR | rfecv | 15.4463 | 2.8928 | 18.0 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` |

## austres (série trivial — 1 lag candidato)

| Família | Método | RMSE | MAPE | NFeatures | Fonte |
|---|---|---|---|---|---|
| ARIMA | sem Feature Selection | 18.9243 | 0.0784 | — | `results/baseline_metrics.csv` |
| MLP | sem Feature Selection | 34.3456 | 0.1864 | 1.0 | `results/baseline_metrics.csv` |
| SVR | sem Feature Selection | **13.9128** | 0.0588 | 1.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | sem Feature Selection | 19.0858 | 0.0800 | 1.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | f_test | 19.1103 | 0.0805 | 1.0 | `results/chamados_v4_fs_ftest/metrics.csv` |
| ARIMA-MLP | mutual_info | 19.0331 | 0.0798 | 1.0 | `results/chamados_v4_fs_mutualinfo/metrics.csv` |
| ARIMA-MLP | rf_embedded | 18.9874 | 0.0793 | 1.0 | `results/chamados_v4_fs_rfembedded/metrics.csv` |
| ARIMA-MLP | lasso | 19.0764 | 0.0798 | 1.0 | `results/chamados_v4_fs_lasso/metrics.csv` |
| ARIMA-MLP | rfecv | 19.1592 | 0.0808 | 1.0 | `results/chamados_v4_fs_rfecv/metrics.csv` |
| ARIMA-SVR | sem Feature Selection | 18.9119 | 0.0815 | 1.0 | `results/baseline_metrics.csv` |
| ARIMA-SVR | f_test | 18.9119 | 0.0815 | 1.0 | `results/chamados_v4_fs_arimasvr_ftest/metrics.csv` |
| ARIMA-SVR | mutual_info | 18.9119 | 0.0815 | 1.0 | `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv` |
| ARIMA-SVR | rf_embedded | 18.9119 | 0.0815 | 1.0 | `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv` |
| ARIMA-SVR | lasso | 18.9119 | 0.0815 | 1.0 | `results/chamados_v4_fs_arimasvr_lasso/metrics.csv` |
| ARIMA-SVR | rfecv | 18.9119 | 0.0815 | 1.0 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` |

## coloradoRiver

| Família | Método | RMSE | MAPE | NFeatures | Fonte |
|---|---|---|---|---|---|
| ARIMA | sem Feature Selection | 0.3240 | 28.8172 | — | `results/baseline_metrics.csv` |
| MLP | sem Feature Selection | 0.2248 | 26.0964 | 16.0 | `results/baseline_metrics.csv` |
| SVR | sem Feature Selection | **0.1266** | 12.7649 | 16.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | sem Feature Selection | 0.3287 | 29.1131 | 16.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | f_test | 0.3252 | 28.8510 | 9.0 | `results/chamados_v4_fs_ftest/metrics.csv` |
| ARIMA-MLP | mutual_info | 0.3290 | 29.1654 | 15.0 | `results/chamados_v4_fs_mutualinfo/metrics.csv` |
| ARIMA-MLP | rf_embedded | 0.3260 | 28.8307 | 4.0 | `results/chamados_v4_fs_rfembedded/metrics.csv` |
| ARIMA-MLP | lasso | 0.3279 | 29.1065 | 1.0 | `results/chamados_v4_fs_lasso/metrics.csv` |
| ARIMA-MLP | rfecv | 0.3277 | 29.0808 | 11.2 | `results/chamados_v4_fs_rfecv/metrics.csv` |
| ARIMA-SVR | sem Feature Selection | 0.3370 | 30.0020 | 16.0 | `results/baseline_metrics.csv` |
| ARIMA-SVR | f_test | 0.3199 | 28.3806 | 9.0 | `results/chamados_v4_fs_arimasvr_ftest/metrics.csv` |
| ARIMA-SVR | mutual_info | 0.2987 | 27.1720 | 1.0 | `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv` |
| ARIMA-SVR | rf_embedded | 0.2971 | 27.5856 | 4.0 | `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv` |
| ARIMA-SVR | lasso | 0.3285 | 29.0110 | 1.0 | `results/chamados_v4_fs_arimasvr_lasso/metrics.csv` |
| ARIMA-SVR | rfecv | 0.3080 | 27.9191 | 9.0 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` |

## sunspot

| Família | Método | RMSE | MAPE | NFeatures | Fonte |
|---|---|---|---|---|---|
| ARIMA | sem Feature Selection | 19.1175 | 36.5031 | — | `results/baseline_metrics.csv` |
| MLP | sem Feature Selection | **17.0825** | 34.9428 | 9.0 | `results/baseline_metrics.csv` |
| SVR | sem Feature Selection | 20.3491 | 32.8312 | 9.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | sem Feature Selection | 19.0730 | 40.2544 | 9.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | f_test | 19.1010 | 39.5355 | 5.0 | `results/chamados_v4_fs_ftest/metrics.csv` |
| ARIMA-MLP | mutual_info | 19.0921 | 40.3311 | 9.0 | `results/chamados_v4_fs_mutualinfo/metrics.csv` |
| ARIMA-MLP | rf_embedded | 18.9510 | 39.4229 | 4.0 | `results/chamados_v4_fs_rfembedded/metrics.csv` |
| ARIMA-MLP | lasso | 18.9487 | 39.4989 | 2.0 | `results/chamados_v4_fs_lasso/metrics.csv` |
| ARIMA-MLP | rfecv | 19.0241 | 40.1889 | 6.4 | `results/chamados_v4_fs_rfecv/metrics.csv` |
| ARIMA-SVR | sem Feature Selection | 19.2650 | 43.1953 | 9.0 | `results/baseline_metrics.csv` |
| ARIMA-SVR | f_test | 20.2858 | 48.2318 | 1.0 | `results/chamados_v4_fs_arimasvr_ftest/metrics.csv` |
| ARIMA-SVR | mutual_info | 19.2650 | 43.1953 | 9.0 | `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv` |
| ARIMA-SVR | rf_embedded | 20.4808 | 46.0397 | 4.0 | `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv` |
| ARIMA-SVR | lasso | 21.0606 | 50.5677 | 2.0 | `results/chamados_v4_fs_arimasvr_lasso/metrics.csv` |
| ARIMA-SVR | rfecv | 18.9894 | 40.4742 | 4.0 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` |
