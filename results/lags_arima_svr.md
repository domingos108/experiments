# ARIMA-SVR com Additive — lags selecionados x RMSE

Documento gerado mecanicamente a partir de `results/lag_selection_consensus_v1.csv` e `results/benchmark_master_with_sources_v1.csv` (Tarefa 9). Para métodos estocásticos (`rf_embedded`/`rfecv` em famílias com `model_exec=10`), o conjunto de lags reportado é o conjunto modal (mais frequente entre as 10 repetições) — a frequência exata está indicada entre parênteses. Nenhuma conclusão foi adicionada.


## ARIMA-SVR


### airlines

**ARIMA-SVR sem Feature Selection**
- Utilizou todas as 20 lags candidatas
- RMSE: 15.1665 (fonte: `results/baseline_metrics.csv`)

**ARIMA-SVR com Feature Selection — `ftest`**
- Selecionou 20 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_20,lag_19,lag_18,lag_17,lag_16,lag_15,lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 15.1665 (fonte: `results/chamados_v4_fs_arimasvr_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_ftest/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `mutualinfo`**
- Selecionou 20 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_20,lag_19,lag_18,lag_17,lag_16,lag_15,lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 15.1665 (fonte: `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_mutualinfo/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `rfembedded`**
- Selecionou 9 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_20,lag_18,lag_16,lag_13,lag_12,lag_7,lag_6,lag_4,lag_2
- RMSE: 20.0391 (fonte: `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_rfembedded/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `lasso`**
- Selecionou 1 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_20
- RMSE: 19.2861 (fonte: `results/chamados_v4_fs_arimasvr_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_lasso/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `rfecv`**
- Selecionou 18 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_20,lag_19,lag_18,lag_17,lag_16,lag_15,lag_13,lag_12,lag_11,lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 15.4463 (fonte: `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_rfecv/selected_features_detail.csv`

### austres (trivial — 1 lag candidato)

**ARIMA-SVR sem Feature Selection**
- Utilizou todas as 1 lags candidatas
- RMSE: 18.9119 (fonte: `results/baseline_metrics.csv`)

**ARIMA-SVR com Feature Selection — `ftest`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 18.9119 (fonte: `results/chamados_v4_fs_arimasvr_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_ftest/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `mutualinfo`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 18.9119 (fonte: `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_mutualinfo/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `rfembedded`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 18.9119 (fonte: `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_rfembedded/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `lasso`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 18.9119 (fonte: `results/chamados_v4_fs_arimasvr_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_lasso/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `rfecv`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 18.9119 (fonte: `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_rfecv/selected_features_detail.csv`

### coloradoRiver

**ARIMA-SVR sem Feature Selection**
- Utilizou todas as 16 lags candidatas
- RMSE: 0.336957 (fonte: `results/baseline_metrics.csv`)

**ARIMA-SVR com Feature Selection — `ftest`**
- Selecionou 9 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_16,lag_13,lag_11,lag_10,lag_9,lag_8,lag_6,lag_3,lag_2
- RMSE: 0.319876 (fonte: `results/chamados_v4_fs_arimasvr_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_ftest/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `mutualinfo`**
- Selecionou 1 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_12
- RMSE: 0.298722 (fonte: `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_mutualinfo/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `rfembedded`**
- Selecionou 4 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_13,lag_12,lag_11,lag_1
- RMSE: 0.297139 (fonte: `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_rfembedded/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `lasso`**
- Selecionou 1 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_11
- RMSE: 0.328528 (fonte: `results/chamados_v4_fs_arimasvr_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_lasso/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `rfecv`**
- Selecionou 9 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_3,lag_2,lag_1
- RMSE: 0.308007 (fonte: `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_rfecv/selected_features_detail.csv`

### sunspot

**ARIMA-SVR sem Feature Selection**
- Utilizou todas as 9 lags candidatas
- RMSE: 19.265 (fonte: `results/baseline_metrics.csv`)

**ARIMA-SVR com Feature Selection — `ftest`**
- Selecionou 1 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9
- RMSE: 20.2858 (fonte: `results/chamados_v4_fs_arimasvr_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_ftest/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `mutualinfo`**
- Selecionou 9 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 19.265 (fonte: `results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_mutualinfo/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `rfembedded`**
- Selecionou 4 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9,lag_4,lag_2,lag_1
- RMSE: 20.4808 (fonte: `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_rfembedded/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `lasso`**
- Selecionou 2 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9,lag_4
- RMSE: 21.0606 (fonte: `results/chamados_v4_fs_arimasvr_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_lasso/selected_features_detail.csv`

**ARIMA-SVR com Feature Selection — `rfecv`**
- Selecionou 4 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9,lag_4,lag_2,lag_1
- RMSE: 18.9894 (fonte: `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_arimasvr_rfecv/selected_features_detail.csv`