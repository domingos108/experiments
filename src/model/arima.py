import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from input import input
import config
from metrics import metrics
from model import generics

from sklearn.base import BaseEstimator
from sklearn.base import clone

import pmdarima as pm

import warnings

# This will suppress ALL FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# apenas para garantir execução resultados antigos
class ResultExp:
    def __init__(self, metrics_results):
        self.metrics_results = metrics_results
class Arima(BaseEstimator):

    def __init__(self, seazonal_lag, horizon=1):
        self.seazonal_lag = seazonal_lag
        self.is_ts_mode = True
        self.horizon = horizon

    def fit(self, ts):
        self.forecaster = pm.auto_arima(ts,
            start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5,
            start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2,
            max_order=5, m= self.seazonal_lag, 
            stepwise=True, trace=True, maxiter=30
        ) 

    def predict_steps(self, ts_test):
        train_predicted = self.forecaster.predict_in_sample()

        prevs_h_steps = []
        
        qtd_prevs = ts_test.shape[0]
        for t in ts_test:
            prevs_h_steps.append(self.forecaster.predict(self.horizon)[self.horizon-1])
            self.forecaster.update(t)
            qtd_prevs = qtd_prevs - 1
   
        if self.horizon>1:

            return train_predicted, prevs_h_steps[0:-(self.horizon-1)]
        
        return train_predicted, prevs_h_steps
    
def exec_training_testing(base_name, experiment_id, model_name, seazonal_lag, horizon, force=True):

    fold, title = generics.format_names(experiment_id, base_name, f'{horizon}{model_name}')

    if generics.file_exists(title) and (not force):
        print('Modelo já executado')
        return None
    
    normalize = False
    diff_kpss = False
    lag_size = 'auto'
    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE,
        'horizon': horizon,
        'lag_size': lag_size,
        'diff_kpss': diff_kpss,
        'normalize': normalize,
        'type_filter': None
    }
    base_info = input.open_format_train_val_test(base_name, exec_config)
    
    (
        ts_univariate,
        df_train, 
        df_val,
        df_test, 
        min_max_scaler,
        test_size, 
        val_size,
        is_stationary, 
        original_ts,
        _
    ) = base_info.sequential_return()

    model = Arima(seazonal_lag, horizon)
    forecaster = clone(model).set_params(** {})
    forecaster.fit(original_ts[0:-(test_size+val_size + horizon-1)])
    
    train_predict, test_predict = forecaster.predict_steps(original_ts[-(test_size +val_size + horizon-1):])

    test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict[-test_size:])

    time_exec = {'testing': None, 'training': None}
    
    if val_size>0:
        val_predict = test_predict[0:-test_size]
    else:
        val_predict = None

    result_dict = {
        'train_predict': train_predict, 
        'val_predict': val_predict, 
        'test_predict':test_predict[-test_size:],
        'val_metrics': None,
        'test_metrics': test_metrics,
        'time_exec': time_exec,
        'params': None,
        'best_metric': None
    }
    result = generics.ResultExp(result_dict)
    generics.save_result(fold, title, [{'experiment': result, 'val_metric': None}])


def ma_filter(MA, error_values):
    filtred = np.zeros(shape=[len(error_values)])

    values = error_values

    for i in range(MA-1, len(values)):
        soma = 0
        for j in range(0, MA):
            soma = soma + values[i - j]

        filtred[i]=(soma / MA)

    return filtred[(MA-1):]

def fit_kurtosis(values):
    return np.abs(3 - kurtosis(values))

def find_best_ma(ts_train):
    k_list = []
    v_max = 12
    range_values = list(range(2, v_max))
    for i in range_values:
        filtered = ma_filter(i,ts_train)
    
        k_list.append(fit_kurtosis(filtered))

    best_k = range_values[np.argmin(k_list)]

    return best_k

def exec_marima_training_testing(base_name, experiment_id, model_name, seazonal_lag, force=True, horizon = 1):
    fold, title = generics.format_names(experiment_id, base_name, model_name)

    if generics.file_exists(title) and (not force):
        print('Modelo já executado')
        return None
    
    normalize = False
    diff_kpss = False
    lag_size = None
    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": 0,
        'horizon': horizon
    }
    base_info = input.open_format_train_val_test(base_name, normalize, lag_size, exec_config, diff_kpss)
    
    (
        ts_univariate,
        df_train, 
        df_val,
        df_test, 
        min_max_scaler,
        test_size, 
        val_size,
        is_stationary, 
        original_ts,
        _
    ) = base_info.sequential_return()

    best_ma = find_best_ma(original_ts[0:-test_size].copy())
    print(f"MA SELECTED {best_ma}")

    ma_ts = ma_filter(best_ma, original_ts) 
    residual_ts = ma_ts - original_ts[best_ma-1:]

    model = Arima(1)
    forecaster = clone(model).set_params(** {})
    forecaster.fit(ma_ts[0:-test_size])

    train_predict, test_predict = forecaster.predict_steps(ma_ts[-test_size:])
    test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict)
    time_exec = {'testing': None, 'training': None}
 
    result_dict = {
        'train_predict': train_predict, 
        'val_predict': None, 
        'test_predict':test_predict,
        'val_metrics': None,
        'test_metrics': test_metrics,
        'time_exec': time_exec,
        'params': None,
        'best_metric': None,
        'residual_series': residual_ts

    }
    result = generics.ResultExp(result_dict)
    generics.save_result(fold, title, [{'experiment': result, 'val_metric': None}])



def exec_ma_training_testing(base_name, experiment_id, model_name, seazonal_lag, horizon, force=True, lag_size='auto'):

    fold, title = generics.format_names(experiment_id, base_name, f'{horizon}{model_name}')

    if generics.file_exists(title) and (not force):
        print('Modelo já executado')
        return None
    
    normalize = False
    diff_kpss = False
    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE,
        'horizon': horizon,
        'lag_size': lag_size,
        'diff_kpss': diff_kpss,
        'normalize': normalize,
        'type_filter': None
    }
    base_info = input.open_format_train_val_test(base_name, exec_config)
    
    (
        ts_univariate,
        df_train, 
        df_val,
        df_test, 
        min_max_scaler,
        test_size, 
        val_size,
        is_stationary, 
        original_ts,
        _
    ) = base_info.sequential_return()

    train_predict= df_train.drop(columns = ['actual']).mean(axis=1)
    test_predict = df_test.drop(columns = ['actual']).mean(axis=1)

    test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict[-test_size:])
    time_exec = {'testing': None, 'training': None}
    
    if val_size>0:
        val_predict = df_val.drop(columns = ['actual']).mean(axis=1)
    else:
        val_predict = None

    result_dict = {
        'train_predict': train_predict, 
        'val_predict': val_predict, 
        'test_predict':test_predict[-test_size:],
        'val_metrics': None,
        'test_metrics': test_metrics,
        'time_exec': time_exec,
        'params': None,
        'best_metric': None
    }
    result = generics.ResultExp(result_dict)
    generics.save_result(fold, title, [{'experiment': result, 'val_metric': None}])


def predict_steps(ts_test, horizon, arima):
    train_predicted = arima.predict_in_sample()
    
    prevs_h_steps = []
    
    qtd_prevs = ts_test.shape[0]
    for t in ts_test:
        prevs_h_steps.append(arima.predict(horizon)[horizon-1])
        arima.update(t)
        qtd_prevs = qtd_prevs - 1
    
    if horizon>1:
    
        return train_predicted, prevs_h_steps[0:-(horizon-1)]
    
    return train_predicted, prevs_h_steps


def exec_training_testing_valid_fits(base_name, experiment_id, model_name, seazonal_lag, horizon, force=True, lag_size='auto'):
    fold, title = generics.format_names(experiment_id, base_name, f'{horizon}{model_name}')
    if generics.file_exists(title) and (not force):
        print('Modelo já executado')
        return None
        
    normalize = False
    diff_kpss = False
    lag_size = 'auto'

    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE,
        'horizon': horizon,
        'lag_size': lag_size,
        'diff_kpss': diff_kpss,
        'normalize': normalize,
        'type_filter': None
    }
    base_info = input.open_format_train_val_test(base_name, exec_config)
    all_ts = base_info.original_ts
    test_size = base_info.test_size
    val_size = base_info.val_size
    ts_train = all_ts[0:-(test_size+val_size + horizon-1)]
    ts_test = all_ts[-(test_size +val_size + horizon-1):]
    
    model_auto = pm.auto_arima(ts_train,
            start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5,
            start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2,
            max_order=5, m = seazonal_lag, 
            stepwise=True, trace=True, maxiter=30, return_valid_fits= True
    ) 

    pv_train = pd.DataFrame()
    pv_test = pd.DataFrame()
    aic_models = []
    print(f"ALL VALID FITS {len(model_auto)}")
    for i, arima in enumerate(model_auto):
        print(arima.order)
        if sum(arima.seasonal_order + arima.order) > 0:
            aic_models.append({"model_name":f'arima_{i}', 'aic': arima.aic()})
            train_predicted, prevs_h_steps = predict_steps(ts_test, horizon, arima)
            pv_train[f'arima_{i}'] = train_predicted
            pv_test[f'arima_{i}'] = prevs_h_steps
           
   
    ordered_cols = pd.DataFrame(aic_models).sort_values('aic')['model_name'].to_list()
    df_prevs = pd.concat(
        [pv_train[ordered_cols], pv_test[ordered_cols]]
    )
    error_series = df_prevs.rsub(all_ts, axis=0)

    mean_test =  pv_test[ordered_cols[0]].values

    test_metrics = metrics.gerenerate_metric_results(
        all_ts[-base_info.test_size:], 
        mean_test[-base_info.test_size:]
    )

    result_dict = {
        'train_predict': pv_train[ordered_cols[0]].values, 
        'val_predict': None, 
        'test_predict': pv_test[ordered_cols[0]].values,
        'val_metrics': None,
        'test_metrics': test_metrics,
        'time_exec': None,
        'params': None,
        'best_metric': None
    }

    result = generics.ResultExp(result_dict)

    result.df_prevs = df_prevs
    result.all_ts = all_ts
    result.error_series = error_series

    generics.save_result(fold, title, [{'experiment': result, 'val_metric': None}])