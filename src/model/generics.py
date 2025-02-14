import pickle as pkl
from pathlib import Path
import os

from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing


from input import input
from metrics import metrics
import config


def fit_predict_ml_schemma(model,x_train, y_train, x_val, x_test):
    model.fit(x_train, y_train)
    test_predict = model.predict(x_test)
    val_predict = np.array([])
    if x_val.shape[0]>0:
        val_predict = model.predict(x_val)

    train_predict = model.predict(x_train)
    return train_predict, val_predict, test_predict

def fit_predict_ts_schemma(model, ts_univariate, lag_size, test_size, val_size, min_max_scaler):

    ts_univariate_norm = min_max_scaler.transform(ts_univariate.reshape(-1, 1)).flatten()
    ts_train = ts_univariate_norm[0:-(test_size+val_size)].copy()
    model.fit(ts_train, lag_size)

    prevs = model.predict_steps(ts_univariate_norm)

    train_predict = prevs[0:-(test_size+val_size)]
    val_predict = prevs[-(test_size+val_size): -test_size]
    test_predict = prevs[-test_size:]
  
    return train_predict, val_predict, test_predict


def fit_predict_model(model, base_name, normalize, lag_size, exec_config):
    (
        ts_univariate,
        df_train, 
        df_val,
        df_test, 
        min_max_scaler,
        test_size, 
        val_size
    ) = input.open_format_train_val_test(base_name, normalize, lag_size, exec_config)
    
    y_train = df_train['actual'].values
    x_train = df_train.drop(columns=['actual']).values

    y_val = df_val['actual'].values
    x_val = df_val.drop(columns=['actual']).values

    y_test = df_test['actual'].values
    x_test = df_test.drop(columns=['actual']).values 

    if model.__dict__.get('is_ts_mode', False):

        (
            train_predict, 
            val_predict, 
            test_predict
        ) = fit_predict_ts_schemma(model, ts_univariate, lag_size, test_size, val_size, min_max_scaler)
      
    else:
        (
            train_predict, 
            val_predict, 
            test_predict
        ) = fit_predict_ml_schemma(model,x_train, y_train, x_val, x_test)

    if normalize:
        test_predict = min_max_scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
        train_predict = min_max_scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
        y_test = min_max_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_train = min_max_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        if y_val.shape[0] > 0:
            val_predict = min_max_scaler.inverse_transform(val_predict.reshape(-1, 1)).flatten()
            y_val = min_max_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    test_metrics = metrics.gerenerate_metric_results(y_test, test_predict)

    if y_val.shape[0]==0:
        val_metrics={}
    else:
        val_metrics = metrics.gerenerate_metric_results(y_val, val_predict)

    return {
        'train_predict': train_predict, 
        'val_predict': val_predict, 
        'test_predict':test_predict,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
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


def grid_seach(model,  base_name, normalize, lag_size, parameters,
              model_exec, model_name, experiment_id,
              group_metrics_name = 'val_metrics', metric = 'RMSE', force = True):
    
    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE
    }

    fold, title = format_names(experiment_id, base_name, model_name)

    if file_exists(title) and (not force):
        print('Modelo já executado')
        return None

    list_params=list(ParameterGrid(parameters))

    list_metrics = []

    model_exec = 1 if model_exec < 1 else model_exec
   
    for params_actual in tqdm(list_params):

        local_metrics = []
        for _ in range(0, model_exec): 
            forecaster = clone(model).set_params(** params_actual)
            result = fit_predict_model(forecaster, base_name, normalize, lag_size, exec_config)
            result['params'] = params_actual
            local_metrics.append(result[group_metrics_name][metric])

        list_metrics.append(np.mean(local_metrics))

    int_arg_min = np.argmin(list_metrics)

    predict_results = predict_test_set(model, base_name, normalize, 
                                       lag_size, model_exec,
                                       list_params[int_arg_min])
    save_result(fold, title, predict_results)


def predict_test_set(model, base_name, normalize, lag_size, model_exec, best_params):
    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": 0
    }
    list_exec = []
    for k in range(0, model_exec):
        forecaster = clone(model).set_params(** best_params)
        result = fit_predict_model(forecaster, base_name, normalize, lag_size, exec_config)
        result['params'] = best_params
        list_exec.append(result)
    
    return list_exec

def grid_seach_multiple_bases(model,  base_name_list, normalize, lag_size_list, parameters, model_exec, model_name, experiment_id):
    for base_name,lag_size in zip(base_name_list, lag_size_list):
        print(base_name)
        grid_seach(model,  base_name, 
                   normalize, 
                   lag_size, 
                   parameters, model_exec,
                   model_name, experiment_id)