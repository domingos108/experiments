# Melhor e pior desempenho por série

Documento gerado mecanicamente a partir de `results/benchmark_master_with_sources_v1.csv` (Tarefa 9). Cada linha reporta a melhor e a pior combinação (Família × Método de FS, incluindo `sem Feature Selection` e o baseline `ARIMA`) dentre as 25 combinações avaliadas para aquela série, para cada métrica disponível. Nenhuma conclusão foi adicionada — apenas os números e o arquivo de origem de cada um.

Convenção de direção: MSE/RMSE/MAE/MAPE/theil/ARV — menor é melhor. IA/POCID — maior é melhor.


## airlines

| Métrica | Melhor combo | Melhor valor | Fonte (melhor) | Pior combo | Pior valor | Fonte (pior) |
|---|---|---|---|---|---|---|
| MSE | ARIMA-SVR (sem Feature Selection) | 230.024 | `results/baseline_metrics.csv` | MLP (sem Feature Selection) | 785.158 | `results/baseline_metrics.csv` |
| RMSE | ARIMA-SVR (sem Feature Selection) | 15.1665 | `results/baseline_metrics.csv` | MLP (sem Feature Selection) | 27.6977 | `results/baseline_metrics.csv` |
| MAE | ARIMA-SVR + rfecv | 13.2355 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` | MLP (sem Feature Selection) | 23.4683 | `results/baseline_metrics.csv` |
| MAPE | ARIMA-SVR + rfecv | 2.89278 | `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv` | MLP (sem Feature Selection) | 5.10915 | `results/baseline_metrics.csv` |
| theil | ARIMA-SVR (sem Feature Selection) | 0.089602 | `results/baseline_metrics.csv` | MLP (sem Feature Selection) | 0.418761 | `results/baseline_metrics.csv` |
| ARV | ARIMA-SVR (sem Feature Selection) | 0.039462 | `results/baseline_metrics.csv` | MLP (sem Feature Selection) | 0.127838 | `results/baseline_metrics.csv` |
| IA | ARIMA-SVR (sem Feature Selection) | 0.990068 | `results/baseline_metrics.csv` | MLP (sem Feature Selection) | 0.96615 | `results/baseline_metrics.csv` |
| POCID | ARIMA-SVR (sem Feature Selection) | 92.8571 | `results/baseline_metrics.csv` | SVR + lasso | 71.4286 | `results/chamados_v4_fs_svr_lasso/metrics.csv` |

## austres (série trivial — ver nota)

> **Nota:** esta série tem apenas 1 lag candidato via `lag_size='auto'` — não há espaço real de seleção de features. Os números abaixo são reportados por completude, mas não devem ser usados para comparar eficácia entre métodos de FS.


| Métrica | Melhor combo | Melhor valor | Fonte (melhor) | Pior combo | Pior valor | Fonte (pior) |
|---|---|---|---|---|---|---|
| MSE | SVR (sem Feature Selection) | 193.565 | `results/baseline_metrics.csv` | MLP + mutualinfo | 2112.79 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| RMSE | SVR (sem Feature Selection) | 13.9128 | `results/baseline_metrics.csv` | MLP + mutualinfo | 42.0051 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| MAE | SVR (sem Feature Selection) | 10.3144 | `results/baseline_metrics.csv` | MLP + mutualinfo | 40.5294 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| MAPE | SVR (sem Feature Selection) | 0.058814 | `results/baseline_metrics.csv` | MLP + mutualinfo | 0.231405 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| theil | SVR (sem Feature Selection) | 0.08107 | `results/baseline_metrics.csv` | MLP + mutualinfo | 1.18319 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| ARV | SVR (sem Feature Selection) | 0.018572 | `results/baseline_metrics.csv` | MLP + mutualinfo | 0.162722 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| IA | SVR (sem Feature Selection) | 0.995194 | `results/baseline_metrics.csv` | MLP + mutualinfo | 0.950669 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| POCID | MLP (sem Feature Selection) | 87.5 | `results/baseline_metrics.csv` | ARIMA (sem Feature Selection) | 75 | `results/baseline_metrics.csv` |

## coloradoRiver

| Métrica | Melhor combo | Melhor valor | Fonte (melhor) | Pior combo | Pior valor | Fonte (pior) |
|---|---|---|---|---|---|---|
| MSE | SVR + lasso | 0.015925 | `results/chamados_v4_fs_svr_lasso/metrics.csv` | ARIMA-SVR (sem Feature Selection) | 0.11354 | `results/baseline_metrics.csv` |
| RMSE | SVR + lasso | 0.126195 | `results/chamados_v4_fs_svr_lasso/metrics.csv` | ARIMA-SVR (sem Feature Selection) | 0.336957 | `results/baseline_metrics.csv` |
| MAE | SVR + lasso | 0.099813 | `results/chamados_v4_fs_svr_lasso/metrics.csv` | ARIMA-SVR (sem Feature Selection) | 0.274324 | `results/baseline_metrics.csv` |
| MAPE | SVR + lasso | 12.6728 | `results/chamados_v4_fs_svr_lasso/metrics.csv` | ARIMA-SVR (sem Feature Selection) | 30.002 | `results/baseline_metrics.csv` |
| theil | SVR + ftest | 0.579074 | `results/chamados_v4_fs_svr_ftest/metrics.csv` | MLP + mutualinfo | 3.1588 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| ARV | SVR + ftest | 0.543095 | `results/chamados_v4_fs_svr_ftest/metrics.csv` | MLP + mutualinfo | 1.19672 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` |
| IA | SVR + lasso | 0.878237 | `results/chamados_v4_fs_svr_lasso/metrics.csv` | ARIMA-SVR (sem Feature Selection) | 0.654406 | `results/baseline_metrics.csv` |
| POCID | MLP (sem Feature Selection) | 71.0811 | `results/baseline_metrics.csv` | ARIMA-SVR (sem Feature Selection) | 48.6486 | `results/baseline_metrics.csv` |

## sunspot

| Métrica | Melhor combo | Melhor valor | Fonte (melhor) | Pior combo | Pior valor | Fonte (pior) |
|---|---|---|---|---|---|---|
| MSE | MLP + mutualinfo | 290.546 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` | SVR + lasso | 953.399 | `results/chamados_v4_fs_svr_lasso/metrics.csv` |
| RMSE | MLP + mutualinfo | 17.037 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` | SVR + lasso | 30.8772 | `results/chamados_v4_fs_svr_lasso/metrics.csv` |
| MAE | MLP + lasso | 13.5458 | `results/chamados_v4_fs_mlp_lasso/metrics.csv` | SVR + lasso | 22.1617 | `results/chamados_v4_fs_svr_lasso/metrics.csv` |
| MAPE | SVR (sem Feature Selection) | 32.8312 | `results/baseline_metrics.csv` | SVR + lasso | 52.9133 | `results/chamados_v4_fs_svr_lasso/metrics.csv` |
| theil | ARIMA-SVR + rfembedded | 0.325414 | `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv` | SVR + rfembedded | 1.46232 | `results/chamados_v4_fs_svr_rfembedded/metrics.csv` |
| ARV | MLP + mutualinfo | 0.166905 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` | SVR + rfembedded | 0.591202 | `results/chamados_v4_fs_svr_rfembedded/metrics.csv` |
| IA | MLP + mutualinfo | 0.96002 | `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv` | SVR + lasso | 0.868373 | `results/chamados_v4_fs_svr_lasso/metrics.csv` |
| POCID | SVR (sem Feature Selection) | 82.1429 | `results/baseline_metrics.csv` | ARIMA-MLP (sem Feature Selection) | 60.7143 | `results/baseline_metrics.csv` |


## Perguntas para investigação futura (não respondidas aqui, apenas levantadas pelos dados)
- Em quantas séries o `ARIMA` puro (sem FS, sem híbrido) aparece como melhor ou pior combo em alguma métrica?
- Existe alguma série onde o melhor e o pior combo pertencem à mesma família (ex. dois métodos de FS diferentes na mesma família, um no topo e outro no fundo)?
- As métricas concordam entre si sobre qual é o melhor combo, ou RMSE e MAPE apontam para vencedores diferentes em alguma série?
