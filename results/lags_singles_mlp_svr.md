# MLP e SVR (single) — lags selecionados x RMSE

Documento gerado mecanicamente a partir de `results/lag_selection_consensus_v1.csv` e `results/benchmark_master_with_sources_v1.csv` (Tarefa 9). Para métodos estocásticos (`rf_embedded`/`rfecv` em famílias com `model_exec=10`), o conjunto de lags reportado é o conjunto modal (mais frequente entre as 10 repetições) — a frequência exata está indicada entre parênteses. Nenhuma conclusão foi adicionada.


## MLP


### airlines

**MLP sem Feature Selection**
- Utilizou todas as 20 lags candidatas
- RMSE: 27.6977 (fonte: `results/baseline_metrics.csv`)

**MLP com Feature Selection — `ftest`**
- Selecionou 5 lag(s) (de 20 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_14,lag_13,lag_12,lag_11,lag_1
- RMSE: 21.6136 (fonte: `results/chamados_v4_fs_mlp_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_ftest/selected_features_detail.csv`

**MLP com Feature Selection — `mutualinfo`**
- Selecionou 9 lag(s) (de 20 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_3,lag_2,lag_1
- RMSE: 23.5213 (fonte: `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_mutualinfo/selected_features_detail.csv`

**MLP com Feature Selection — `rfembedded`**
- Selecionou 1 lag(s) (de 20 candidatas), conjunto observado em 8/10 das repetições
- Lags: lag_12
- RMSE: 22.6379 (fonte: `results/chamados_v4_fs_mlp_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_rfembedded/selected_features_detail.csv`

**MLP com Feature Selection — `lasso`**
- Selecionou 14 lag(s) (de 20 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_18,lag_16,lag_14,lag_13,lag_12,lag_11,lag_10,lag_8,lag_7,lag_6,lag_5,lag_3,lag_2,lag_1
- RMSE: 26.4854 (fonte: `results/chamados_v4_fs_mlp_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_lasso/selected_features_detail.csv`

**MLP com Feature Selection — `rfecv`**
- Selecionou 3 lag(s) (de 20 candidatas), conjunto observado em 4/10 das repetições
- Lags: lag_12,lag_11,lag_1
- RMSE: 22.9617 (fonte: `results/chamados_v4_fs_mlp_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_rfecv/selected_features_detail.csv`

### austres (trivial — 1 lag candidato)

**MLP sem Feature Selection**
- Utilizou todas as 1 lags candidatas
- RMSE: 34.3456 (fonte: `results/baseline_metrics.csv`)

**MLP com Feature Selection — `ftest`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 40.5727 (fonte: `results/chamados_v4_fs_mlp_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_ftest/selected_features_detail.csv`

**MLP com Feature Selection — `mutualinfo`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 42.0051 (fonte: `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_mutualinfo/selected_features_detail.csv`

**MLP com Feature Selection — `rfembedded`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 29.6767 (fonte: `results/chamados_v4_fs_mlp_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_rfembedded/selected_features_detail.csv`

**MLP com Feature Selection — `lasso`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 29.9106 (fonte: `results/chamados_v4_fs_mlp_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_lasso/selected_features_detail.csv`

**MLP com Feature Selection — `rfecv`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 32.7449 (fonte: `results/chamados_v4_fs_mlp_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_rfecv/selected_features_detail.csv`

### coloradoRiver

**MLP sem Feature Selection**
- Utilizou todas as 16 lags candidatas
- RMSE: 0.224816 (fonte: `results/baseline_metrics.csv`)

**MLP com Feature Selection — `ftest`**
- Selecionou 9 lag(s) (de 16 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_16,lag_13,lag_12,lag_11,lag_8,lag_7,lag_6,lag_5,lag_1
- RMSE: 0.149509 (fonte: `results/chamados_v4_fs_mlp_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_ftest/selected_features_detail.csv`

**MLP com Feature Selection — `mutualinfo`**
- Selecionou 1 lag(s) (de 16 candidatas), conjunto observado em 7/10 das repetições
- Lags: lag_1
- RMSE: 0.225898 (fonte: `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_mutualinfo/selected_features_detail.csv`

**MLP com Feature Selection — `rfembedded`**
- Selecionou 2 lag(s) (de 16 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_12,lag_1
- RMSE: 0.134073 (fonte: `results/chamados_v4_fs_mlp_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_rfembedded/selected_features_detail.csv`

**MLP com Feature Selection — `lasso`**
- Selecionou 7 lag(s) (de 16 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_14,lag_12,lag_11,lag_9,lag_7,lag_6,lag_1
- RMSE: 0.179973 (fonte: `results/chamados_v4_fs_mlp_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_lasso/selected_features_detail.csv`

**MLP com Feature Selection — `rfecv`**
- Selecionou 3 lag(s) (de 16 candidatas), conjunto observado em 4/10 das repetições
- Lags: lag_16,lag_12,lag_1
- RMSE: 0.15964 (fonte: `results/chamados_v4_fs_mlp_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_rfecv/selected_features_detail.csv`

### sunspot

**MLP sem Feature Selection**
- Utilizou todas as 9 lags candidatas
- RMSE: 17.0825 (fonte: `results/baseline_metrics.csv`)

**MLP com Feature Selection — `ftest`**
- Selecionou 9 lag(s) (de 9 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 17.0825 (fonte: `results/chamados_v4_fs_mlp_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_ftest/selected_features_detail.csv`

**MLP com Feature Selection — `mutualinfo`**
- Selecionou 9 lag(s) (de 9 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 17.037 (fonte: `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_mutualinfo/selected_features_detail.csv`

**MLP com Feature Selection — `rfembedded`**
- Selecionou 1 lag(s) (de 9 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_1
- RMSE: 27.4817 (fonte: `results/chamados_v4_fs_mlp_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_rfembedded/selected_features_detail.csv`

**MLP com Feature Selection — `lasso`**
- Selecionou 6 lag(s) (de 9 candidatas), conjunto observado em 10/10 das repetições
- Lags: lag_9,lag_8,lag_5,lag_3,lag_2,lag_1
- RMSE: 17.1025 (fonte: `results/chamados_v4_fs_mlp_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_lasso/selected_features_detail.csv`

**MLP com Feature Selection — `rfecv`**
- Selecionou 9 lag(s) (de 9 candidatas), conjunto observado em 3/10 das repetições
- Lags: lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 17.2995 (fonte: `results/chamados_v4_fs_mlp_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_mlp_rfecv/selected_features_detail.csv`

## SVR


### airlines

**SVR sem Feature Selection**
- Utilizou todas as 20 lags candidatas
- RMSE: 19.3867 (fonte: `results/baseline_metrics.csv`)

**SVR com Feature Selection — `ftest`**
- Selecionou 20 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_20,lag_19,lag_18,lag_17,lag_16,lag_15,lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 19.3867 (fonte: `results/chamados_v4_fs_svr_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_ftest/selected_features_detail.csv`

**SVR com Feature Selection — `mutualinfo`**
- Selecionou 20 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_20,lag_19,lag_18,lag_17,lag_16,lag_15,lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 19.3867 (fonte: `results/chamados_v4_fs_svr_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_mutualinfo/selected_features_detail.csv`

**SVR com Feature Selection — `rfembedded`**
- Selecionou 1 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_12
- RMSE: 22.0753 (fonte: `results/chamados_v4_fs_svr_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_rfembedded/selected_features_detail.csv`

**SVR com Feature Selection — `lasso`**
- Selecionou 7 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_13,lag_12,lag_11,lag_10,lag_8,lag_7,lag_5
- RMSE: 22.4213 (fonte: `results/chamados_v4_fs_svr_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_lasso/selected_features_detail.csv`

**SVR com Feature Selection — `rfecv`**
- Selecionou 2 lag(s) (de 20 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_15,lag_12
- RMSE: 26.8775 (fonte: `results/chamados_v4_fs_svr_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_rfecv/selected_features_detail.csv`

### austres (trivial — 1 lag candidato)

**SVR sem Feature Selection**
- Utilizou todas as 1 lags candidatas
- RMSE: 13.9128 (fonte: `results/baseline_metrics.csv`)

**SVR com Feature Selection — `ftest`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 13.9128 (fonte: `results/chamados_v4_fs_svr_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_ftest/selected_features_detail.csv`

**SVR com Feature Selection — `mutualinfo`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 13.9128 (fonte: `results/chamados_v4_fs_svr_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_mutualinfo/selected_features_detail.csv`

**SVR com Feature Selection — `rfembedded`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 13.9128 (fonte: `results/chamados_v4_fs_svr_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_rfembedded/selected_features_detail.csv`

**SVR com Feature Selection — `lasso`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 13.9128 (fonte: `results/chamados_v4_fs_svr_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_lasso/selected_features_detail.csv`

**SVR com Feature Selection — `rfecv`**
- Selecionou 1 lag(s) (de 1 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 13.9128 (fonte: `results/chamados_v4_fs_svr_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_rfecv/selected_features_detail.csv`

### coloradoRiver

**SVR sem Feature Selection**
- Utilizou todas as 16 lags candidatas
- RMSE: 0.126589 (fonte: `results/baseline_metrics.csv`)

**SVR com Feature Selection — `ftest`**
- Selecionou 5 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_14,lag_13,lag_12,lag_3,lag_2
- RMSE: 0.140534 (fonte: `results/chamados_v4_fs_svr_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_ftest/selected_features_detail.csv`

**SVR com Feature Selection — `mutualinfo`**
- Selecionou 9 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_15,lag_14,lag_12,lag_10,lag_7,lag_5,lag_4,lag_3,lag_2
- RMSE: 0.156187 (fonte: `results/chamados_v4_fs_svr_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_mutualinfo/selected_features_detail.csv`

**SVR com Feature Selection — `rfembedded`**
- Selecionou 4 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_12,lag_10,lag_2,lag_1
- RMSE: 0.145685 (fonte: `results/chamados_v4_fs_svr_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_rfembedded/selected_features_detail.csv`

**SVR com Feature Selection — `lasso`**
- Selecionou 15 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_15,lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 0.126195 (fonte: `results/chamados_v4_fs_svr_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_lasso/selected_features_detail.csv`

**SVR com Feature Selection — `rfecv`**
- Selecionou 14 lag(s) (de 16 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_15,lag_14,lag_13,lag_12,lag_11,lag_10,lag_9,lag_8,lag_7,lag_6,lag_4,lag_3,lag_2,lag_1
- RMSE: 0.150212 (fonte: `results/chamados_v4_fs_svr_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_rfecv/selected_features_detail.csv`

### sunspot

**SVR sem Feature Selection**
- Utilizou todas as 9 lags candidatas
- RMSE: 20.3491 (fonte: `results/baseline_metrics.csv`)

**SVR com Feature Selection — `ftest`**
- Selecionou 5 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9,lag_6,lag_5,lag_2,lag_1
- RMSE: 25.2998 (fonte: `results/chamados_v4_fs_svr_ftest/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_ftest/selected_features_detail.csv`

**SVR com Feature Selection — `mutualinfo`**
- Selecionou 5 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9,lag_6,lag_5,lag_2,lag_1
- RMSE: 25.2998 (fonte: `results/chamados_v4_fs_svr_mutualinfo/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_mutualinfo/selected_features_detail.csv`

**SVR com Feature Selection — `rfembedded`**
- Selecionou 1 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_1
- RMSE: 27.3136 (fonte: `results/chamados_v4_fs_svr_rfembedded/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_rfembedded/selected_features_detail.csv`

**SVR com Feature Selection — `lasso`**
- Selecionou 6 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9,lag_8,lag_5,lag_3,lag_2,lag_1
- RMSE: 30.8772 (fonte: `results/chamados_v4_fs_svr_lasso/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_lasso/selected_features_detail.csv`

**SVR com Feature Selection — `rfecv`**
- Selecionou 8 lag(s) (de 9 candidatas), conjunto observado em 1/1 das repetições
- Lags: lag_9,lag_8,lag_7,lag_5,lag_4,lag_3,lag_2,lag_1
- RMSE: 20.1886 (fonte: `results/chamados_v4_fs_svr_rfecv/metrics.csv`)
- Fonte da seleção: `results/chamados_v4_fs_svr_rfecv/selected_features_detail.csv`