import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from input import input
from metrics import metrics


def inverse_difference(last_ob: Union[np.ndarray, List[float]], value: Union[np.ndarray, List[float]]) -> Union[np.ndarray, List[float]]:
    """Add the last observed value back to a differenced series.

    Args:
        last_ob (Union[np.ndarray, List[float]]): Previous observed values.
        value (Union[np.ndarray, List[float]]): Differenced values.

    Returns:
        Union[np.ndarray, List[float]]: Reconstructed values.
    """
    return value + last_ob


def get_inverse_difference(last_ob: Union[np.ndarray, List[float]], value: Union[np.ndarray, List[float]]) -> List[Union[float, np.floating]]:
    """Apply inverse differencing elementwise to a series.

    Args:
        last_ob (Union[np.ndarray, List[float]]): Previous observed values.
        value (Union[np.ndarray, List[float]]): Differenced values.

    Returns:
        List[Union[float, np.floating]]: Restored values.
    """
    return [inverse_difference(last_ob[i], value[i]) for i in range(len(value))]


def fit_predict_ml_schemma(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Fit a scikit-learn model and return train, validation and test predictions.

    Args:
        model (Any): Scikit-learn compatible model.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        x_val (np.ndarray): Validation features.
        x_test (np.ndarray): Test features.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]: Predictions and execution timings.
    """
    start_time_training = time.time()
    model.fit(x_train, y_train)
    end_time_training = time.time()

    start_time_test = time.time()
    test_predict = model.predict(x_test)
    end_time_test = time.time()
    val_predict = np.array([])
    if x_val.shape[0] > 0:
        val_predict = model.predict(x_val)

    train_predict = model.predict(x_train)

    time_exec = {
        'training': end_time_training - start_time_training,
        'testing': end_time_test - start_time_test,
    }

    return train_predict, val_predict, test_predict, time_exec


class SKlearnModel:
    """Wrapper for training and evaluating scikit-learn forecasting models."""

    def __init__(
        self,
        model: Any,
        experiment_id: str,
        base_name: str,
        model_name: str,
        force: bool = True,
        normalize: bool = True,
        experiment_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the wrapper with model and experiment configuration.

        Args:
            model (Any): Scikit-learn-compatible model.
            experiment_id (str): Identifier of the experiment.
            base_name (str): Base series name.
            model_name (str): Name of the model.
            force (bool, optional): Whether to overwrite existing results. Defaults to True.
            normalize (bool, optional): Whether to normalize inputs. Defaults to True.
            experiment_params (Optional[Dict[str, Any]], optional): Experiment configuration. Defaults to None.
        """
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params or {}
        self.force = force
        self.normalize = normalize

    def fit_predict(self) -> None:
        """Train the model, generate predictions and store metric results."""
        lag_size_base = self.experiment_params['lag_size']
        horizon = self.experiment_params.get('horizon', 1)
        type_filter = self.experiment_params.get('type_filter', None)
        normalize = self.normalize
        diff_kpss = self.experiment_params.get('diff_kpss', True)

        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size'],
            'horizon': horizon,
            'lag_size': lag_size_base,
            'diff_kpss': diff_kpss,
            'normalize': normalize,
            'type_filter': type_filter,
        }

        base_info = input.open_format_train_val_test(self.base_name, exec_config)
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
            _,
        ) = base_info.sequential_return()

        y_train = df_train['actual'].values
        x_train = df_train.drop(columns=['actual']).values

        y_val = df_val['actual'].values
        x_val = df_val.drop(columns=['actual']).values

        x_test = df_test.drop(columns=['actual']).values

        (
            train_predict,
            val_predict,
            test_predict,
            time_exec,
        ) = fit_predict_ml_schemma(self.model, x_train, y_train, x_val, x_test)

        y_train_original = original_ts[0:-(test_size + val_size)][-len(train_predict):]
        y_test_original = original_ts[-test_size:]
        y_val_original = original_ts[-(test_size + val_size):-test_size]

        if normalize:
            test_predict = min_max_scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
            train_predict = min_max_scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
            if y_val.shape[0] > 0:
                val_predict = min_max_scaler.inverse_transform(val_predict.reshape(-1, 1)).flatten()

        if (is_stationary is False) and (diff_kpss is True):
            train_predict = get_inverse_difference(original_ts[0:-(test_size + val_size + horizon)], train_predict)

            if y_val.shape[0] > 0:
                val_predict = get_inverse_difference(original_ts[-(test_size + val_size + horizon):-(test_size + horizon)], val_predict)

            test_predict = get_inverse_difference(original_ts[-(test_size + horizon):-horizon], test_predict)

        test_metrics = metrics.gerenerate_metric_results(y_test_original, test_predict)

        if y_val.shape[0] == 0:
            val_metrics = {}
        else:
            val_metrics = metrics.gerenerate_metric_results(y_val_original, val_predict)

        self.metrics_results = {
            'train_predict': train_predict,
            'val_predict': val_predict,
            'test_predict': test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': time_exec,
        }