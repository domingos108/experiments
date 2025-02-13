import pickle as pkl
from pathlib import Path
import os

from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from tqdm import tqdm
import numpy as np

from input import input
from metrics import metrics
import config


def fit_predict_model(model, base_name, normalize, lag_size):
    (df_train, 
    df_val,
    df_test, 
    min_max_scaler) = input.open_format_train_val_test(base_name, normalize, lag_size)

    y_train = df_train['actual'].values
    x_train = df_train.drop(columns=['actual'])

    y_val = df_val['actual'].values
    x_val = df_val.drop(columns=['actual'])

    y_test = df_test['actual'].values
    x_test = df_test.drop(columns=['actual'])

    model.fit(x_train, y_train)

    test_predict = model.predict(x_test)
    val_predict = model.predict(x_val)
    train_predict = model.predict(x_train)

    if normalize:
        test_predict = min_max_scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
        val_predict = min_max_scaler.inverse_transform(val_predict.reshape(-1, 1)).flatten()
        train_predict = min_max_scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()

        y_test = min_max_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_val = min_max_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_train = min_max_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

    test_metrics = metrics.gerenerate_metric_results(y_test, test_predict)
    val_metrics = metrics.gerenerate_metric_results(y_val, val_predict)
    train_metrics = metrics.gerenerate_metric_results(y_train, train_predict)
    
    return {
        'train_predict': train_predict, 
        'val_predict': val_predict, 
        'test_predict':test_predict,
        'train_metrics': train_metrics,
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
              group_metrics_name = 'val_metrics', metric = 'RMSE'):

    fold, title = format_names(experiment_id, base_name, model_name)

    if file_exists(title):
        print('Modelo já executado')
        return None

    list_params=list(ParameterGrid(parameters))

    list_metrics = []
    list_results = []

    model_exec = 1 if model_exec < 1 else model_exec
   
    for params_actual in tqdm(list_params):
        forecaster = clone(model).set_params(** params_actual)
        local_metrics = []
        local_results = []
        for _ in range(0, model_exec): 
            result = fit_predict_model(model, base_name, normalize, lag_size)
            result['forecaster'] = forecaster
            local_metrics.append(result[group_metrics_name][metric])
            local_results.append(result)

        list_metrics.append(np.mean(local_metrics))
        list_results.append(local_results)

    int_arg_min = np.argmin(list_metrics)

    save_result(fold, title, list_results[int_arg_min])

def grid_seach_multiple_bases(model,  base_name_list, normalize, lag_size_list, parameters, model_exec, model_name, experiment_id):
    for base_name,lag_size in zip(base_name_list, lag_size_list):
        print(base_name)
        grid_seach(model,  base_name, 
                   normalize, 
                   lag_size, 
                   parameters, model_exec,
                   model_name, experiment_id)