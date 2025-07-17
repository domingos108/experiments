from glob import glob 

import numpy as np
import pandas as pd

from model import generics
from input import input


def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

def root_mean_square_error(y_true, y_pred):

    return mean_square_error(y_true, y_pred)**0.5

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(np.where(y_true == 0)[0]) > 0:
        return np.inf
    else:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs(y_true - y_pred))


def u_theil(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])).sum()

    return error_sup / error_inf


def average_relative_variance(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred, mean)).sum()

    return error_sup / error_inf


def index_agreement(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.abs(np.subtract(y_true, y_pred))).sum()

    error_inf = np.abs(np.subtract(y_pred, mean)) + np.abs(np.subtract(y_true, mean))
    error_inf = np.square(error_inf).sum()

    return 1 - (error_sup / error_inf)


def prediction_of_change_in_direction(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    true_sub = np.subtract(y_true[0:(len(y_true) - 1)], y_true[1:(len(y_true))])
    pred_sub = np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])

    mult = true_sub * pred_sub
    result = 0
    for m in mult:
        if m > 0:
            result = result + 1

    return (100 * (result / len(y_true)))


def gerenerate_metric_results(y_true, y_pred):
    return {'MSE': mean_square_error(y_true, y_pred),
            'RMSE':root_mean_square_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'theil': u_theil(y_true, y_pred),
            'ARV': average_relative_variance(y_true, y_pred),
            'IA': index_agreement(y_true, y_pred),
            'POCID': prediction_of_change_in_direction(y_true, y_pred)}

def get_best_test_forecast(metric, exec_pkl, model_name, serie_name):
    val_metrics = [ep['test_metrics'][metric] for ep in exec_pkl]
    forecast_values = [ep['test_predict'] for ep in exec_pkl]
    forecast_values = forecast_values[np.argmin(val_metrics)]

    return pd.DataFrame({'prev':forecast_values, 
                        'model': model_name,
                        'ts': serie_name,
                        'id': range(0, len(forecast_values)),
                        })

def get_real_values(df_prevs):

    df_info_ts = (df_prevs.groupby('ts')['id'].max() ).reset_index().copy()
    df_info_ts['id'] = df_info_ts['id'].values  +1
    #df_info_ts = (df_prevs.groupby('ts')['id'].max() +1).reset_index()

    for base_info in df_info_ts.to_dict('records'):
        df = input.load_raw_data(base_info['ts'])
        real_test = df[-base_info['id']:].reset_index(drop=True)['y']
        df_real = pd.DataFrame({
            'prev': real_test,
            'model': 'real',
            'ts': base_info['ts'],
            'id': range(0, base_info['id']) 
        })
        
        df_prevs = pd.concat([df_prevs, df_real])

    return df_prevs
def open_fold_result(experiment_id,  group_metrics_name = 'val_metrics', metric = 'RMSE'):
    fold, _ = generics.format_names(experiment_id, '', '')
    exec_model = []
    df_all_metrics = pd.DataFrame()
    df_prevs = pd.DataFrame()
    for pth in glob(fold+'*'):
        model_name = pth.split('_')[-1].split('.pk')[0]
        serie_name = pth.split('/')[-1].split('_')[0]
        exec_pkl = generics.open_saved_result(pth)
        exec_model.append(exec_pkl)

        all_metrics = []
        for ep in exec_pkl:
            dict_temp = ep['test_metrics']
            test_timing = ep['time_exec']['testing'] if  ep['time_exec']['testing'] else np.inf
            training__time = ep['time_exec']['training'] if ep['time_exec']['training'] else np.inf
            best_metric = ep['best_metric'] if ep['best_metric'] else np.inf

            dict_temp['val_metric'] = best_metric
            dict_temp['time_testing'] = test_timing
            dict_temp['time_training'] = training__time
            all_metrics.append(dict_temp)
        
        df_metric = pd.DataFrame(all_metrics)
        df_metric['model'] = model_name
        df_metric['ts'] = serie_name
        
        df_all_metrics = pd.concat([df_all_metrics, df_metric])

        df_prevs = pd.concat([
            df_prevs,
            get_best_test_forecast(metric, exec_pkl, model_name, serie_name)
        ])

    df_mean_metrics = df_all_metrics.groupby(['ts', 'model']).mean()
    df_prevs = get_real_values(df_prevs)
    return df_mean_metrics, df_all_metrics, df_prevs
