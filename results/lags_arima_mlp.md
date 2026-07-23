# ARIMA-MLP com Additive — lags selecionados x RMSE

Documento gerado mecanicamente a partir de `results/lag_selection_consensus_v1.csv` e `results/benchmark_master_with_sources_v1.csv` (Tarefa 9). Para métodos estocásticos (`rf_embedded`/`rfecv` em famílias com `model_exec=10`), o conjunto de lags reportado é o conjunto modal (mais frequente entre as 10 repetições) — a frequência exata está indicada entre parênteses. Nenhuma conclusão foi adicionada.


## ARIMA-MLP


### airlines

**ARIMA-MLP sem Feature Selection**
- Utilizou todas as 20 lags candidatas
- RMSE: 17.4362 (fonte: `results/baseline_metrics.csv`)

**ARIMA-MLP com Feature Selection — `ftest`**
- Selecionou 5 lag(s) (de 20 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_20,lag_16,lag_13,lag_2,lag_1
- RMSE: 19.1139 (fonte: `results/chamados_v4_fs_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_ftest/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `mutualinfo`**
- Selecionou 9 lag(s) (de 20 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_19,lag_18,lag_17,lag_14,lag_12,lag_5,lag_4,lag_2,lag_1
- RMSE: 19.6141 (fonte: `results/chamados_v4_fs_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mutualinfo/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `rfembedded`**
- Selecionou 8 lag(s) (de 20 candidatas), conjunto observado em 2/10 das repetições
- Lags: lag_20,lag_18,lag_13,lag_12,lag_7,lag_6,lag_4,lag_2
- RMSE: 19.4301 (fonte: `results/chamados_v4_fs_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_rfembedded/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `lasso`**
- Selecionou 1 lag(s) (de 20 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_20
- RMSE: 19.7328 (fonte: `results/chamados_v4_fs_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_lasso/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `rfecv`**
- Selecionou 20 lag(s) (de 20 candidatas), conjunto observado em 4/10 das repetições
- Lags: lag_20,lag_19,lag_18,lag_17,lag_16,lag_15,lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 18.0167 (fonte: `results/chamados_v4_fs_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_rfecv/selected_features_detail.csv`

### austres (trivial — 1 lag candidato)

**ARIMA-MLP sem Feature Selection**
- Utilizou todas as 1 lags candidatas
- RMSE: 19.0858 (fonte: `results/baseline_metrics.csv`)

**ARIMA-MLP com Feature Selection — `ftest`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 19.1103 (fonte: `results/chamados_v4_fs_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_ftest/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `mutualinfo`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 19.0331 (fonte: `results/chamados_v4_fs_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mutualinfo/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `rfembedded`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 18.9874 (fonte: `results/chamados_v4_fs_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_rfembedded/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `lasso`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 19.0764 (fonte: `results/chamados_v4_fs_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_lasso/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `rfecv`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 19.1592 (fonte: `results/chamados_v4_fs_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_rfecv/selected_features_detail.csv`

### coloradoRiver

**ARIMA-MLP sem Feature Selection**
- Utilizou todas as 16 lags candidatas
- RMSE: 0.328696 (fonte: `results/baseline_metrics.csv`)

**ARIMA-MLP com Feature Selection — `ftest`**
- Selecionou 9 lag(s) (de 16 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_16,lag_13,lag_11,lag_10,lag_9,lag_8,lag_6,lag_3,lag_2
- RMSE: 0.325178 (fonte: `results/chamados_v4_fs_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_ftest/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `mutualinfo`**
- Selecionou 15 lag(s) (de 16 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_16,lag_15,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 0.329002 (fonte: `results/chamados_v4_fs_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mutualinfo/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `rfembedded`**
- Selecionou 4 lag(s) (de 16 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_13,lag_12,lag_11,lag_1
- RMSE: 0.326045 (fonte: `results/chamados_v4_fs_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_rfembedded/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `lasso`**
- Selecionou 1 lag(s) (de 16 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_11
- RMSE: 0.327909 (fonte: `results/chamados_v4_fs_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_lasso/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `rfecv`**
- Selecionou 14 lag(s) (de 16 candidatas), conjunto observado em 1/10 das repetições
- Lags: lag_15,lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_5,lag_3,lag_2,lag_1
- RMSE: 0.327655 (fonte: `results/chamados_v4_fs_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_rfecv/selected_features_detail.csv`

### sunspot

**ARIMA-MLP sem Feature Selection**
- Utilizou todas as 9 lags candidatas
- RMSE: 19.073 (fonte: `results/baseline_metrics.csv`)

**ARIMA-MLP com Feature Selection — `ftest`**
- Selecionou 5 lag(s) (de 9 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_9,lag_8,lag_5,lag_4,lag_3
- RMSE: 19.101 (fonte: `results/chamados_v4_fs_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_ftest/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `mutualinfo`**
- Selecionou 9 lag(s) (de 9 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 19.0921 (fonte: `results/chamados_v4_fs_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mutualinfo/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `rfembedded`**
- Selecionou 4 lag(s) (de 9 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_9,lag_4,lag_2,lag_1
- RMSE: 18.951 (fonte: `results/chamados_v4_fs_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_rfembedded/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `lasso`**
- Selecionou 2 lag(s) (de 9 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_9,lag_4
- RMSE: 18.9487 (fonte: `results/chamados_v4_fs_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_lasso/selected_features_detail.csv`

**ARIMA-MLP com Feature Selection — `rfecv`**
- Selecionou 8 lag(s) (de 9 candidatas), conjunto observado em 3/10 das repetições
- Lags: lag_9,lag_8,lag_7,lag_6,lag_4,lag_3,lag_2,lag_1
- RMSE: 19.0241 (fonte: `results/chamados_v4_fs_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_rfecv/selected_features_detail.csv`