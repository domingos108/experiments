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
| ARIMA-MLP | f_test | 19.1156 | 3.2214 | 5.0 | `results/chamados_v4_fs_ftest/metrics.csv` |
| ARIMA-MLP | mutual_info | 19.6759 | 3.4620 | 9.0 | `results/chamados_v4_fs_mutualinfo/metrics.csv` |
| ARIMA-MLP | rf_embedded | 19.4033 | 3.3032 | 7.6 | `results/chamados_v4_fs_rfembedded/metrics.csv` |
| ARIMA-MLP | lasso | 19.7365 | 3.2824 | 1.0 | `results/chamados_v4_fs_lasso/metrics.csv` |
| ARIMA-MLP | rfecv | 18.3069 | 3.2908 | 14.9 | `results/chamados_v4_fs_rfecv/metrics.csv` |
| ARIMA-SVR | sem Feature Selection | 15.1665 | 3.0674 | 20.0 | `results/baseline_metrics.csv` |
| ARIMA-SVR | f_test | 15.1665 | 3.0674 | 20.0 | `results/chamados_v4_fs_arimasvr_ftest/metrics.csv` |
| ARIMA-SVR | mutual_info | 15.1665 | 3.0674 | 20.0 | `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv` |
| ARIMA-SVR | rf_embedded | 19.8481 | 3.4086 | 8.0 | `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv` |
| ARIMA-SVR | lasso | 19.2861 | 3.1099 | 1.0 | `results/chamados_v4_fs_arimasvr_lasso/metrics.csv` |
| ARIMA-SVR | rfecv | **14.3877** | 2.5826 | 17.0 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` |

## austres (série trivial — 1 lag candidato)

| Família | Método | RMSE | MAPE | NFeatures | Fonte |
|---|---|---|---|---|---|
| ARIMA | sem Feature Selection | 18.9243 | 0.0784 | — | `results/baseline_metrics.csv` |
| MLP | sem Feature Selection | 34.3456 | 0.1864 | 1.0 | `results/baseline_metrics.csv` |
| SVR | sem Feature Selection | **13.9128** | 0.0588 | 1.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | sem Feature Selection | 19.0858 | 0.0800 | 1.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | f_test | 19.0515 | 0.0801 | 1.0 | `results/chamados_v4_fs_ftest/metrics.csv` |
| ARIMA-MLP | mutual_info | 19.0515 | 0.0801 | 1.0 | `results/chamados_v4_fs_mutualinfo/metrics.csv` |
| ARIMA-MLP | rf_embedded | 19.0515 | 0.0801 | 1.0 | `results/chamados_v4_fs_rfembedded/metrics.csv` |
| ARIMA-MLP | lasso | 19.0515 | 0.0801 | 1.0 | `results/chamados_v4_fs_lasso/metrics.csv` |
| ARIMA-MLP | rfecv | 19.0515 | 0.0801 | 1.0 | `results/chamados_v4_fs_rfecv/metrics.csv` |
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
| ARIMA-MLP | f_test | 0.3270 | 29.0087 | 15.0 | `results/chamados_v4_fs_ftest/metrics.csv` |
| ARIMA-MLP | mutual_info | 0.3306 | 29.2489 | 16.0 | `results/chamados_v4_fs_mutualinfo/metrics.csv` |
| ARIMA-MLP | rf_embedded | 0.3284 | 29.0761 | 4.0 | `results/chamados_v4_fs_rfembedded/metrics.csv` |
| ARIMA-MLP | lasso | 0.3265 | 29.0154 | 1.0 | `results/chamados_v4_fs_lasso/metrics.csv` |
| ARIMA-MLP | rfecv | 0.3252 | 28.8741 | 12.1 | `results/chamados_v4_fs_rfecv/metrics.csv` |
| ARIMA-SVR | sem Feature Selection | 0.3370 | 30.0020 | 16.0 | `results/baseline_metrics.csv` |
| ARIMA-SVR | f_test | 0.3199 | 28.3806 | 9.0 | `results/chamados_v4_fs_arimasvr_ftest/metrics.csv` |
| ARIMA-SVR | mutual_info | 0.2987 | 27.1720 | 1.0 | `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv` |
| ARIMA-SVR | rf_embedded | 0.2971 | 27.5856 | 4.0 | `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv` |
| ARIMA-SVR | lasso | 0.3285 | 29.0110 | 1.0 | `results/chamados_v4_fs_arimasvr_lasso/metrics.csv` |
| ARIMA-SVR | rfecv | 0.3131 | 27.8790 | 11.0 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` |

## sunspot

| Família | Método | RMSE | MAPE | NFeatures | Fonte |
|---|---|---|---|---|---|
| ARIMA | sem Feature Selection | 19.1175 | 36.5031 | — | `results/baseline_metrics.csv` |
| MLP | sem Feature Selection | **17.0825** | 34.9428 | 9.0 | `results/baseline_metrics.csv` |
| SVR | sem Feature Selection | 20.3491 | 32.8312 | 9.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | sem Feature Selection | 19.0730 | 40.2544 | 9.0 | `results/baseline_metrics.csv` |
| ARIMA-MLP | f_test | 19.1160 | 39.8119 | 5.0 | `results/chamados_v4_fs_ftest/metrics.csv` |
| ARIMA-MLP | mutual_info | 19.1165 | 40.5758 | 9.0 | `results/chamados_v4_fs_mutualinfo/metrics.csv` |
| ARIMA-MLP | rf_embedded | 18.9216 | 39.3073 | 4.0 | `results/chamados_v4_fs_rfembedded/metrics.csv` |
| ARIMA-MLP | lasso | 18.9496 | 39.6235 | 2.0 | `results/chamados_v4_fs_lasso/metrics.csv` |
| ARIMA-MLP | rfecv | 19.0106 | 39.6631 | 5.3 | `results/chamados_v4_fs_rfecv/metrics.csv` |
| ARIMA-SVR | sem Feature Selection | 19.2650 | 43.1953 | 9.0 | `results/baseline_metrics.csv` |
| ARIMA-SVR | f_test | 20.2858 | 48.2318 | 1.0 | `results/chamados_v4_fs_arimasvr_ftest/metrics.csv` |
| ARIMA-SVR | mutual_info | 19.2650 | 43.1953 | 9.0 | `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv` |
| ARIMA-SVR | rf_embedded | 20.4808 | 46.0397 | 4.0 | `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv` |
| ARIMA-SVR | lasso | 21.0606 | 50.5677 | 2.0 | `results/chamados_v4_fs_arimasvr_lasso/metrics.csv` |
| ARIMA-SVR | rfecv | 19.4332 | 42.4394 | 6.0 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` |
