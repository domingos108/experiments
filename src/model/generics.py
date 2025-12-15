import pickle as pkl
from pathlib import Path
import os
import time

from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from tqdm.auto import tqdm
import numpy as np

from input import input
from metrics import metrics
import config


def fit_predict_ml_schemma(model,x_train, y_train, x_val, x_test):
    start_time_training = time.time()
    model.fit(x_train, y_train)
    end_time_training = time.time()

    start_time_test = time.time()
    test_predict = model.predict(x_test)
    end_time_test = time.time()
    val_predict = np.array([])
    if x_val.shape[0]>0:
        val_predict = model.predict(x_val)

    train_predict = model.predict(x_train)

    time_exec = {
        'training': end_time_training - start_time_training,
        'testing': end_time_test - start_time_test
    }

    return train_predict, val_predict, test_predict, time_exec

def fit_predict_ts_schemma(model, ts_univariate, test_size, val_size):

    ts_train = ts_univariate[0:-(test_size+val_size)].copy()
    
    model.fit(ts_train)

    prevs = model.predict_steps(ts_univariate)
   
    train_predict = prevs[0:-(test_size+val_size)]
    val_predict = prevs[-(test_size+val_size): -test_size]
    test_predict = prevs[-test_size:]
  
    return train_predict, val_predict, test_predict, {}


def fit_predict_model(model, base_name, normalize, lag_size, exec_config, diff_kpss):
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
    )  = base_info.sequential_return()
    y_train = df_train['actual'].values
    x_train = df_train.drop(columns=['actual']).values

    y_val = df_val['actual'].values
    x_val = df_val.drop(columns=['actual']).values

    x_test = df_test.drop(columns=['actual']).values 
   
    if model.__dict__.get('is_ts_mode', False):

        (
            train_predict, 
            val_predict, 
            test_predict,
            time_exec
        ) = fit_predict_ts_schemma(model, ts_univariate, test_size, val_size)
    
    else:
        (
            train_predict, 
            val_predict, 
            test_predict,
            time_exec
        ) = fit_predict_ml_schemma(model,x_train, y_train, x_val, x_test)
    
    y_train_original = original_ts[0:-(test_size+val_size)][- len(train_predict):]
    y_test_original = original_ts[-test_size:]
    y_val_original = original_ts[-(test_size+val_size): -test_size]

    if normalize:
        test_predict = min_max_scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
        train_predict = min_max_scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
        if y_val.shape[0] > 0:
            val_predict = min_max_scaler.inverse_transform(val_predict.reshape(-1, 1)).flatten()
        
    if (is_stationary is False) and (diff_kpss is True):
        train_predict =  y_train_original + train_predict
        
        if y_val.shape[0] > 0:
            test_predict =  np.concatenate(( [y_val_original[-1]], y_test_original[0:-1])) + test_predict

            val_predict =  np.concatenate(( [y_train_original[-1]], y_val_original[0:-1])) + val_predict
        else:
            test_predict =  np.concatenate(( [y_train_original[-1]], y_test_original[0:-1])) + test_predict

    test_metrics = metrics.gerenerate_metric_results(y_test_original, test_predict)

    if y_val.shape[0]==0:
        val_metrics={}
    else:
        val_metrics = metrics.gerenerate_metric_results(y_val_original, val_predict)

    return {
        'train_predict': train_predict, 
        'val_predict': val_predict, 
        'test_predict':test_predict,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'time_exec': time_exec
    }

def create_path_if_not_exists(path):
    """Creates a directory or nested directories if they don't exist.

    Args:
        path: A string or Path object representing the path to create.
    """
    path_obj = Path(path)  # Convert to Path object if it's a string

    if not path_obj.exists():
        try:
            path_obj.mkdir(parents=True, exist_ok=True)  # parents=True creates parent dirs, exist_ok avoids error if already exists
        except OSError as e:
            print(f"Error creating path {path_obj}: {e}")


def file_exists(file_path):
    """Checks if a file exists using os.path.

    Args:
        file_path: The path to the file (string).

    Returns:
        True if the file exists and is a file, False otherwise.
    """
    return os.path.isfile(file_path)

def open_saved_result(title):
    
    with open(title, 'rb') as handle:
        b = pkl.load(handle)
    return b


def save_result(fold, title, result):

    create_path_if_not_exists(fold)

    with open(title, 'wb') as handle:
        pkl.dump(result, handle)


def format_names(experiment_id, base_name, model_name):
    fold = config.MODEL_DATA_PATH +experiment_id +'/'
    title = fold+base_name.split('.')[0]+'_'+model_name+".pkl"
    return fold, title

def format_forecats(original_ts, 
                    time_exec, 
                    test_size, 
                    val_size, 
                    normalize,
                    min_max_scaler,
                    is_stationary,
                    diff_kpss,
                    y_val,
                    train_predict,
                    val_predict,
                    test_predict

    ):

    y_train_original = original_ts[0:-(test_size+val_size)][- len(train_predict):]
    y_test_original = original_ts[-test_size:]
    y_val_original = original_ts[-(test_size+val_size): -test_size]

    if normalize:
        test_predict = min_max_scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
        train_predict = min_max_scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
        if y_val.shape[0] > 0:
            val_predict = min_max_scaler.inverse_transform(val_predict.reshape(-1, 1)).flatten()
        
    if (is_stationary is False) and (diff_kpss is True):
        train_predict =  y_train_original + train_predict
        
        if y_val.shape[0] > 0:
            test_predict =  np.concatenate(( [y_val_original[-1]], y_test_original[0:-1])) + test_predict

            val_predict =  np.concatenate(( [y_train_original[-1]], y_val_original[0:-1])) + val_predict
        else:
            test_predict =  np.concatenate(( [y_train_original[-1]], y_test_original[0:-1])) + test_predict

    test_metrics = metrics.gerenerate_metric_results(y_test_original, test_predict)

    if y_val.shape[0]==0:
        val_metrics={}
    else:
        val_metrics = metrics.gerenerate_metric_results(y_val_original, val_predict)

    return  {
        'train_predict': train_predict, 
        'val_predict': val_predict, 
        'test_predict':test_predict,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'time_exec': time_exec
    }