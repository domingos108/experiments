"""Helpers for computing forecasting metrics and formatting metric results."""

from glob import glob
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing

from model import generics
from input import input
import config


def mean_square_error(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the mean squared error between expected and predicted values.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The mean squared error between the two series.
    """
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()


def root_mean_square_error(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the root mean squared error between expected and predicted values.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The root mean squared error between the two series.
    """
    return mean_square_error(y_true, y_pred) ** 0.5


def symmetric_mean_absolute_percentage_error(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the symmetric mean absolute percentage error (sMAPE).

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The sMAPE value in percentage terms.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_values = numerator / denominator

    return np.mean(smape_values) * 100


def mean_absolute_percentage_error(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the mean absolute percentage error (MAPE).

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The MAPE value in percentage terms, or infinity if zero values are present.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(np.where(y_true == 0)[0]) > 0:
        return np.inf
    else:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_error(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the mean absolute error between expected and predicted values.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The mean absolute error between the two series.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs(y_true - y_pred))


def u_theil(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the Theil's U statistic for comparing forecasts.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The Theil's U coefficient.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])).sum()

    return error_sup / error_inf


def average_relative_variance(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the average relative variance between expected and predicted values.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The average relative variance coefficient.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred, mean)).sum()

    return error_sup / error_inf


def index_agreement(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the index of agreement between expected and predicted values.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The index of agreement between the two series.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.abs(np.subtract(y_true, y_pred))).sum()

    error_inf = np.abs(np.subtract(y_pred, mean)) + np.abs(np.subtract(y_true, mean))
    error_inf = np.square(error_inf).sum()

    return 1 - (error_sup / error_inf)


def prediction_of_change_in_direction(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> float:
    """Calculate the percentage of correct directional changes between the two series.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        float: The percentage of correctly predicted directional changes.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    true_sub = np.subtract(y_true[0:(len(y_true) - 1)], y_true[1:(len(y_true))])
    pred_sub = np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])

    mult = true_sub * pred_sub
    result = 0
    for m in mult:
        if m > 0:
            result = result + 1

    return 100 * (result / len(y_true))


def gerenerate_metric_results(y_true: Union[np.ndarray, list, pd.Series], y_pred: Union[np.ndarray, list, pd.Series]) -> Dict[str, float]:
    """Compute all supported forecasting metrics for a pair of series.

    Args:
        y_true (Union[np.ndarray, list, pd.Series]): Ground-truth values.
        y_pred (Union[np.ndarray, list, pd.Series]): Predicted values.

    Returns:
        Dict[str, float]: A dictionary containing all metric values.
    """
    return {
        'MSE': mean_square_error(y_true, y_pred),
        'RMSE': root_mean_square_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'SMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'theil': u_theil(y_true, y_pred),
        'ARV': average_relative_variance(y_true, y_pred),
        'IA': index_agreement(y_true, y_pred),
        'POCID': prediction_of_change_in_direction(y_true, y_pred),
    }


def get_best_test_forecast(metric: str, exec_pkl: Any, model_name: str, serie_name: str) -> pd.DataFrame:
    """Select the best forecast from a list of experiment results.

    Args:
        metric (str): Metric name used to select the best forecast.
        exec_pkl (Any): Serialized experiment results.
        model_name (str): Name of the model.
        serie_name (str): Name of the time series.

    Returns:
        pd.DataFrame: A dataframe containing the selected forecast values and metadata.
    """
    val_metrics = [ep['experiment'].metrics_results['test_metrics'][metric] for ep in exec_pkl]
    forecast_values = [ep['experiment'].metrics_results['test_predict'] for ep in exec_pkl]
    forecast_values = forecast_values[np.argmin(val_metrics)]

    return pd.DataFrame({
        'prev': forecast_values,
        'model': model_name,
        'ts': serie_name,
        'id': range(0, len(forecast_values)),
    })


def get_real_values(df_prevs: pd.DataFrame) -> pd.DataFrame:
    """Append the real observed values to the forecast dataframe.

    Args:
        df_prevs (pd.DataFrame): Dataframe containing forecast values and metadata.

    Returns:
        pd.DataFrame: The input dataframe enriched with real values.
    """
    df_info_ts = (df_prevs.groupby('ts')['id'].max()).reset_index().copy()
    df_info_ts['id'] = df_info_ts['id'].values + 1

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


def normalize_results(experiment: Any, serie_name: str) -> Dict[str, Any]:
    """Normalize forecast results using min-max scaling and rebuild metric results.

    Args:
        experiment (Any): Experiment object containing prediction arrays.
        serie_name (str): Name of the time series.

    Returns:
        Dict[str, Any]: A dictionary with normalized metric results.
    """
    train_predict = experiment.metrics_results['train_predict']
    val_predict = experiment.metrics_results['val_predict']
    test_predict = experiment.metrics_results['test_predict']

    prevs = pd.Series(train_predict).to_list() + pd.Series(val_predict).to_list() + pd.Series(test_predict).to_list()

    prevs = np.array(prevs)

    prevs[np.isnan(prevs)] = 0
    prevs[np.isinf(prevs)] = 0

    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE,
        'horizon': 1,
        'lag_size': config.BASE_INFORMATION[serie_name + '.txt']['lag_size'],
        'diff_kpss': False,
        'normalize': True,
        'type_filter': None
    }

    base_info = input.open_format_train_val_test(serie_name + '.txt', exec_config)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))
    min_max_scaler.fit(base_info.original_ts.reshape(-1, 1))
    final_forecast = min_max_scaler.transform(np.array(prevs).reshape(-1, 1)).flatten()

    base_info.original_ts = min_max_scaler.transform(base_info.original_ts.reshape(-1, 1)).flatten()

    return format_metrics_results(final_forecast, base_info)


def open_fold_result(experiment_id: str, group_metrics_name: str = 'val_metrics', metric: str = 'RMSE', normalize: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Open and aggregate results from a fold of experiments.

    Args:
        experiment_id (str): Identifier of the experiment folder.
        group_metrics_name (str): Name of the metrics group to use. Defaults to 'val_metrics'.
        metric (str): Metric used to select the best forecasting result. Defaults to 'RMSE'.
        normalize (bool): Whether to normalize predictions before computing metrics. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Mean metrics, all metrics, and forecast dataframe.
    """
    fold, _ = generics.format_names(experiment_id, '', '')
    exec_model = []
    df_all_metrics = pd.DataFrame()
    df_prevs = pd.DataFrame()

    for pth in glob(fold + '*'):
        try:
            model_name = pth.split('_')[-1].split('.pk')[0]
            serie_name = pth.split('/')[-1].split('_')[0]

            exec_pkl = generics.open_saved_result(pth)

            for p in exec_pkl:
                p['experiment'].model = None

            exec_model.append(exec_pkl)

            all_metrics = []

            for experiment in exec_pkl:
                if normalize:
                    experiment['experiment'].metrics_results = normalize_results(experiment['experiment'], serie_name)

                ep = experiment['experiment'].metrics_results
                dict_temp = ep['test_metrics']
                test_timing = ep['time_exec']['testing'] if ep['time_exec'] else np.inf
                training__time = ep['time_exec']['training'] if ep['time_exec'] else np.inf
                best_metric = experiment['val_metric'] if experiment['val_metric'] else np.inf

                dict_temp['val_metric'] = best_metric
                dict_temp['time_testing'] = test_timing
                dict_temp['time_training'] = training__time
                all_metrics.append(dict_temp)

            df_metric = pd.DataFrame(all_metrics).fillna(-1).infer_objects(copy=False)

            df_metric['model'] = model_name
            df_metric['ts'] = serie_name

            df_all_metrics = pd.concat([df_all_metrics, df_metric])

            df_prevs = pd.concat([
                df_prevs,
                get_best_test_forecast(metric, exec_pkl, model_name, serie_name)
            ])
        except Exception:
            pass

    df_mean_metrics = df_all_metrics.groupby(['ts', 'model']).mean()
    df_prevs = get_real_values(df_prevs)
    return df_mean_metrics, df_all_metrics, df_prevs


def format_metrics_results(final_forecast: np.ndarray, base_info: input.OpenDataOutput) -> Dict[str, Any]:
    """Format forecast arrays into train, validation, and test metric results.

    Args:
        final_forecast (np.ndarray): Forecast values concatenated for train, validation, and test splits.
        base_info (input.OpenDataOutput): Dataset metadata and original series information.

    Returns:
        Dict[str, Any]: A dictionary containing predictions and metric results for each split.
    """
    test_size = base_info.test_size
    val_size = base_info.val_size

    train_predict = final_forecast[0:-(test_size + val_size)]
    val_predict = final_forecast[-(test_size + val_size): -test_size]
    test_predict = final_forecast[-test_size:]

    test_metrics = gerenerate_metric_results(base_info.original_ts[-test_size:], test_predict)

    if val_size is not None and val_size > 0:
        val_metrics = gerenerate_metric_results(base_info.original_ts[-(test_size + val_size): -test_size], val_predict)
    else:
        val_metrics = None

    metrics_results = {
        'train_predict': train_predict,
        'val_predict': val_predict,
        'test_predict': test_predict,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'time_exec': {'testing': None, 'training': None}
    }

    return metrics_results