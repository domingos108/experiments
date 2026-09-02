# Auditoria de Integridade Metodologica -- v1

**Total de checagens: 689** -- ✅ 677 PASS, ❌ 0 FAIL, ⚠️ 0 ATENÇÃO, ➖ 12 N/A

**Nenhuma divergência encontrada.**

## Todas as checagens

| Experimento | Checagem | Status | Detalhe |
|---|---|---|---|
| baseline/1arima/airlines | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1arima/airlines | experiment_params | ➖ N/A | familia sem experiment_params (ex. ARIMA/auto_arima) |
| baseline/1arima/airlines | hyperparameter_parity | ➖ N/A | sem estimador sklearn expondo hiperparametros |
| baseline/1arima/austres | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1arima/austres | experiment_params | ➖ N/A | familia sem experiment_params (ex. ARIMA/auto_arima) |
| baseline/1arima/austres | hyperparameter_parity | ➖ N/A | sem estimador sklearn expondo hiperparametros |
| baseline/1arima/coloradoRiver | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1arima/coloradoRiver | experiment_params | ➖ N/A | familia sem experiment_params (ex. ARIMA/auto_arima) |
| baseline/1arima/coloradoRiver | hyperparameter_parity | ➖ N/A | sem estimador sklearn expondo hiperparametros |
| baseline/1arima/sunspot | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1arima/sunspot | experiment_params | ➖ N/A | familia sem experiment_params (ex. ARIMA/auto_arima) |
| baseline/1arima/sunspot | hyperparameter_parity | ➖ N/A | sem estimador sklearn expondo hiperparametros |
| baseline/1arima/windspeedfortaleza | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1arima/windspeedfortaleza | experiment_params | ➖ N/A | familia sem experiment_params (ex. ARIMA/auto_arima) |
| baseline/1arima/windspeedfortaleza | hyperparameter_parity | ➖ N/A | sem estimador sklearn expondo hiperparametros |
| baseline/1arima/samurec | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1arima/samurec | experiment_params | ➖ N/A | familia sem experiment_params (ex. ARIMA/auto_arima) |
| baseline/1arima/samurec | hyperparameter_parity | ➖ N/A | sem estimador sklearn expondo hiperparametros |
| baseline/1mlp/airlines | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1mlp/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1mlp/airlines | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1mlp/austres | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1mlp/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1mlp/austres | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1mlp/coloradoRiver | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1mlp/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1mlp/coloradoRiver | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1mlp/sunspot | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1mlp/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1mlp/sunspot | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1mlp/windspeedfortaleza | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1mlp/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1mlp/windspeedfortaleza | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1mlp/samurec | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1mlp/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1mlp/samurec | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| mlp/ftest | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| mlp/ftest/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/ftest/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/ftest/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/ftest/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/ftest/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/ftest/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| mlp/ftest/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/ftest/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/ftest/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/ftest/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| mlp/ftest/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/ftest/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/ftest/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/ftest/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| mlp/ftest/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/ftest/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/ftest/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/ftest/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/ftest/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/ftest/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/ftest/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/ftest/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| mlp/ftest/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/ftest/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/mutualinfo | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| mlp/mutualinfo/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/mutualinfo/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/mutualinfo/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/mutualinfo/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/mutualinfo/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/mutualinfo/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| mlp/mutualinfo/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/mutualinfo/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/mutualinfo/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/mutualinfo/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| mlp/mutualinfo/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/mutualinfo/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/mutualinfo/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/mutualinfo/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| mlp/mutualinfo/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/mutualinfo/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/mutualinfo/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/mutualinfo/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/mutualinfo/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/mutualinfo/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/mutualinfo/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/mutualinfo/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| mlp/mutualinfo/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/mutualinfo/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfembedded | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| mlp/rfembedded/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfembedded/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/rfembedded/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfembedded/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfembedded/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfembedded/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| mlp/rfembedded/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfembedded/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfembedded/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfembedded/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| mlp/rfembedded/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfembedded/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfembedded/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfembedded/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| mlp/rfembedded/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfembedded/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfembedded/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfembedded/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/rfembedded/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfembedded/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfembedded/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfembedded/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| mlp/rfembedded/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfembedded/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/lasso | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| mlp/lasso/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/lasso/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/lasso/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/lasso/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/lasso (amostra: airlines) | cv_time_series_split | ✅ PASS | cv=TimeSeriesSplit confirmado por reconstrucao real |
| mlp/lasso/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/lasso/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| mlp/lasso/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/lasso/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/lasso/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/lasso/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| mlp/lasso/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/lasso/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/lasso/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/lasso/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| mlp/lasso/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/lasso/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/lasso/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/lasso/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/lasso/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/lasso/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/lasso/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/lasso/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| mlp/lasso/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/lasso/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfecv | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| mlp/rfecv/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfecv/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/rfecv/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfecv/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfecv (amostra: airlines) | cv_time_series_split | ✅ PASS | cv=TimeSeriesSplit confirmado por reconstrucao real |
| mlp/rfecv/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfecv/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| mlp/rfecv/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfecv/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfecv/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfecv/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| mlp/rfecv/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfecv/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfecv/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfecv/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| mlp/rfecv/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfecv/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfecv/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfecv/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| mlp/rfecv/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfecv/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| mlp/rfecv/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| mlp/rfecv/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| mlp/rfecv/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| mlp/rfecv/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| baseline/1svr/airlines | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1svr/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1svr/airlines | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1svr/austres | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1svr/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1svr/austres | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1svr/coloradoRiver | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1svr/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1svr/coloradoRiver | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1svr/sunspot | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1svr/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1svr/sunspot | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1svr/windspeedfortaleza | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1svr/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1svr/windspeedfortaleza | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1svr/samurec | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1svr/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1svr/samurec | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| svr/ftest | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| svr/ftest/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/ftest/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/ftest/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/ftest/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/ftest/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/ftest/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| svr/ftest/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/ftest/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/ftest/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/ftest/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| svr/ftest/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/ftest/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/ftest/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/ftest/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| svr/ftest/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/ftest/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/ftest/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/ftest/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/ftest/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/ftest/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/ftest/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/ftest/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| svr/ftest/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/ftest/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/mutualinfo | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| svr/mutualinfo/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/mutualinfo/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/mutualinfo/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/mutualinfo/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/mutualinfo/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/mutualinfo/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| svr/mutualinfo/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/mutualinfo/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/mutualinfo/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/mutualinfo/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| svr/mutualinfo/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/mutualinfo/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/mutualinfo/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/mutualinfo/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| svr/mutualinfo/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/mutualinfo/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/mutualinfo/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/mutualinfo/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/mutualinfo/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/mutualinfo/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/mutualinfo/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/mutualinfo/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| svr/mutualinfo/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/mutualinfo/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfembedded | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| svr/rfembedded/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfembedded/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/rfembedded/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfembedded/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfembedded/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfembedded/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| svr/rfembedded/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfembedded/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfembedded/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfembedded/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| svr/rfembedded/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfembedded/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfembedded/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfembedded/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| svr/rfembedded/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfembedded/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfembedded/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfembedded/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/rfembedded/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfembedded/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfembedded/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfembedded/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| svr/rfembedded/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfembedded/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/lasso | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| svr/lasso/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/lasso/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/lasso/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/lasso/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/lasso (amostra: airlines) | cv_time_series_split | ✅ PASS | cv=TimeSeriesSplit confirmado por reconstrucao real |
| svr/lasso/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/lasso/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| svr/lasso/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/lasso/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/lasso/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/lasso/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| svr/lasso/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/lasso/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/lasso/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/lasso/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| svr/lasso/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/lasso/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/lasso/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/lasso/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/lasso/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/lasso/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/lasso/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/lasso/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| svr/lasso/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/lasso/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfecv | pkl_count_and_naming | ✅ PASS | 6 arquivo(s) conferem |
| svr/rfecv/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfecv/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/rfecv/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfecv/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfecv (amostra: airlines) | cv_time_series_split | ✅ PASS | cv=TimeSeriesSplit confirmado por reconstrucao real |
| svr/rfecv/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfecv/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| svr/rfecv/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfecv/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfecv/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfecv/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| svr/rfecv/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfecv/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfecv/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfecv/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| svr/rfecv/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfecv/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfecv/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfecv/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| svr/rfecv/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfecv/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| svr/rfecv/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| svr/rfecv/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| svr/rfecv/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| svr/rfecv/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| baseline/1amv1/airlines | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1amv1/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1amv1/airlines | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1amv1/austres | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1amv1/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1amv1/austres | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1amv1/coloradoRiver | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1amv1/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1amv1/coloradoRiver | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1amv1/sunspot | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1amv1/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1amv1/sunspot | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1amv1/windspeedfortaleza | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1amv1/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1amv1/windspeedfortaleza | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1amv1/samurec | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1amv1/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1amv1/samurec | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| arima_mlp/ftest | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_mlp/ftest/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/ftest/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/ftest/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/ftest/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/ftest/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/ftest/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_mlp/ftest/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/ftest/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/ftest/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/ftest/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_mlp/ftest/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/ftest/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/ftest/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/ftest/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_mlp/ftest/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/ftest/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/ftest/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/ftest/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/ftest/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/ftest/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/ftest/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/ftest/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_mlp/ftest/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/ftest/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/mutualinfo | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_mlp/mutualinfo/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/mutualinfo/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/mutualinfo/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/mutualinfo/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/mutualinfo/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/mutualinfo/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_mlp/mutualinfo/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/mutualinfo/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/mutualinfo/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/mutualinfo/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_mlp/mutualinfo/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/mutualinfo/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/mutualinfo/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/mutualinfo/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_mlp/mutualinfo/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/mutualinfo/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/mutualinfo/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/mutualinfo/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/mutualinfo/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/mutualinfo/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/mutualinfo/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/mutualinfo/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_mlp/mutualinfo/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/mutualinfo/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfembedded | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_mlp/rfembedded/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfembedded/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/rfembedded/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfembedded/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfembedded/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfembedded/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_mlp/rfembedded/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfembedded/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfembedded/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfembedded/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_mlp/rfembedded/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfembedded/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfembedded/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfembedded/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_mlp/rfembedded/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfembedded/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfembedded/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfembedded/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/rfembedded/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfembedded/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfembedded/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfembedded/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_mlp/rfembedded/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfembedded/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/lasso | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_mlp/lasso/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/lasso/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/lasso/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/lasso/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/lasso (amostra: airlines) | cv_time_series_split | ✅ PASS | cv=TimeSeriesSplit confirmado por reconstrucao real |
| arima_mlp/lasso/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/lasso/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_mlp/lasso/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/lasso/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/lasso/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/lasso/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_mlp/lasso/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/lasso/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/lasso/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/lasso/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_mlp/lasso/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/lasso/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/lasso/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/lasso/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/lasso/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/lasso/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/lasso/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/lasso/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_mlp/lasso/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/lasso/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfecv | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_mlp/rfecv/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfecv/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/rfecv/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfecv/airlines | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfecv (amostra: airlines) | cv_time_series_split | ✅ PASS | cv=TimeSeriesSplit confirmado por reconstrucao real |
| arima_mlp/rfecv/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfecv/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_mlp/rfecv/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfecv/austres | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfecv/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfecv/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_mlp/rfecv/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfecv/coloradoRiver | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfecv/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfecv/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_mlp/rfecv/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfecv/sunspot | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfecv/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfecv/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_mlp/rfecv/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfecv/windspeedfortaleza | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| arima_mlp/rfecv/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_mlp/rfecv/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_mlp/rfecv/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_mlp/rfecv/samurec | n_reps | ✅ PASS | 10 repeticao(oes), conforme esperado |
| baseline/1as/airlines | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1as/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1as/airlines | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1as/austres | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1as/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1as/austres | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1as/coloradoRiver | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1as/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1as/coloradoRiver | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1as/sunspot | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1as/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1as/sunspot | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1as/windspeedfortaleza | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1as/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1as/windspeedfortaleza | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| baseline/1as/samurec | baseline_pkl_exists | ✅ PASS | arquivo presente |
| baseline/1as/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| baseline/1as/samurec | hyperparameter_parity | ✅ PASS | hiperparametros fixos conferem |
| arima_svr/ftest | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_svr/ftest/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/ftest/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/ftest/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/ftest/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/ftest/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/ftest/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_svr/ftest/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/ftest/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/ftest/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/ftest/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_svr/ftest/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/ftest/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/ftest/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/ftest/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_svr/ftest/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/ftest/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/ftest/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/ftest/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/ftest/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/ftest/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/ftest/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/ftest/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_svr/ftest/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/ftest/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/mutualinfo | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_svr/mutualinfo/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/mutualinfo/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/mutualinfo/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/mutualinfo/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/mutualinfo/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/mutualinfo/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_svr/mutualinfo/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/mutualinfo/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/mutualinfo/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/mutualinfo/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_svr/mutualinfo/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/mutualinfo/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/mutualinfo/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/mutualinfo/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_svr/mutualinfo/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/mutualinfo/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/mutualinfo/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/mutualinfo/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/mutualinfo/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/mutualinfo/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/mutualinfo/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/mutualinfo/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_svr/mutualinfo/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/mutualinfo/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfembedded | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_svr/rfembedded/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfembedded/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/rfembedded/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfembedded/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfembedded/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfembedded/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_svr/rfembedded/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfembedded/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfembedded/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfembedded/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_svr/rfembedded/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfembedded/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfembedded/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfembedded/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_svr/rfembedded/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfembedded/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfembedded/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfembedded/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/rfembedded/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfembedded/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfembedded/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfembedded/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_svr/rfembedded/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfembedded/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/lasso | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_svr/lasso/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/lasso/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/lasso/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/lasso/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/lasso (amostra: airlines) | cv_time_series_split | ✅ PASS | cv=TimeSeriesSplit confirmado por reconstrucao real |
| arima_svr/lasso/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/lasso/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_svr/lasso/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/lasso/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/lasso/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/lasso/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_svr/lasso/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/lasso/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/lasso/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/lasso/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_svr/lasso/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/lasso/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/lasso/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/lasso/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/lasso/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/lasso/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/lasso/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/lasso/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_svr/lasso/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/lasso/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfecv | pkl_count_and_naming | ✅ PASS | 12 arquivo(s) conferem |
| arima_svr/rfecv/airlines | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfecv/airlines | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/rfecv/airlines | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfecv/airlines | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfecv (amostra: airlines) | cv_time_series_split | ✅ PASS | cv=TimeSeriesSplit confirmado por reconstrucao real |
| arima_svr/rfecv/austres | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfecv/austres | lag_size | ✅ PASS | lag_size=1, conforme RUNBOOK.md |
| arima_svr/rfecv/austres | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfecv/austres | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfecv/coloradoRiver | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfecv/coloradoRiver | lag_size | ✅ PASS | lag_size=16, conforme RUNBOOK.md |
| arima_svr/rfecv/coloradoRiver | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfecv/coloradoRiver | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfecv/sunspot | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfecv/sunspot | lag_size | ✅ PASS | lag_size=9, conforme RUNBOOK.md |
| arima_svr/rfecv/sunspot | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfecv/sunspot | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfecv/windspeedfortaleza | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfecv/windspeedfortaleza | lag_size | ✅ PASS | lag_size=20, conforme RUNBOOK.md |
| arima_svr/rfecv/windspeedfortaleza | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfecv/windspeedfortaleza | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| arima_svr/rfecv/samurec | hyperparameter_parity | ✅ PASS | todos os hiperparametros conferem com o baseline |
| arima_svr/rfecv/samurec | lag_size | ✅ PASS | lag_size=15, conforme RUNBOOK.md |
| arima_svr/rfecv/samurec | experiment_params | ✅ PASS | diff_kpss/horizon/linear_model_name conferem |
| arima_svr/rfecv/samurec | n_reps | ✅ PASS | 1 repeticao(oes), conforme esperado |
| baseline/airlines_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/airlines_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/airlines_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/airlines_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/airlines_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ausbee_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ausbee_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ausbee_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ausbee_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ausbee_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/austres_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/austres_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/austres_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/austres_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/austres_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/coloradoRiver_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/coloradoRiver_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/coloradoRiver_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/coloradoRiver_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/coloradoRiver_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/gasoline_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/gasoline_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/gasoline_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/gasoline_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/gasoline_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/heartrate_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/heartrate_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/heartrate_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/heartrate_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/heartrate_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lakeerie_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lakeerie_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lakeerie_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lakeerie_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lakeerie_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lynx_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lynx_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lynx_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lynx_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/lynx_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/milk_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/milk_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/milk_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/milk_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/milk_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ozon_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ozon_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ozon_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ozon_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/ozon_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/pollution_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/pollution_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/pollution_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/pollution_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/pollution_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/redwine_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/redwine_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/redwine_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/redwine_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/redwine_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/sunspot_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/sunspot_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/sunspot_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/sunspot_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/sunspot_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/temperature_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/temperature_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/temperature_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/temperature_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/temperature_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/Unemployment_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/Unemployment_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/Unemployment_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/Unemployment_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/Unemployment_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/woolyrnq_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/woolyrnq_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/woolyrnq_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/woolyrnq_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/woolyrnq_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/samurec_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/samurec_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/samurec_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/samurec_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/samurec_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/windspeedfortaleza_1arima.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/windspeedfortaleza_1mlp.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/windspeedfortaleza_1svr.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/windspeedfortaleza_1amv1.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| baseline/windspeedfortaleza_1as.pkl | baseline_hash | ✅ PASS | hash confere com a referencia |
| gamma_note | gamma_provisional_note | ✅ PASS | nota provisoria presente e completa |
