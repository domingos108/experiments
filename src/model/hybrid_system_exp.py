import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import clone

from input import input
from metrics import metrics
from model import generics


class ResultExp:
    """Container for experiment metrics and prediction results."""

    def __init__(self, metrics_results: dict) -> None:
        """Initialize the result wrapper.

        Args:
            metrics_results (dict): Dictionary with metric and prediction values.
        """
        self.metrics_results = metrics_results


def invscaling(count: int, power_t: float, learning_rate_init: float) -> float:
    """Compute the inverse scaling factor used in recursive updates.

    Args:
        count (int): Iteration counter.
        power_t (float): Exponent applied to the count.
        learning_rate_init (float): Initial learning rate.

    Returns:
        float: The computed scaling factor.
    """
    return learning_rate_init / np.power(count, power_t)


def input_linear_info(experiment_id: str, base_name: str, experiment_params: dict) -> tuple:
    """Load the linear model forecasts and residuals for hybrid experiments.

    Args:
        experiment_id (str): Identifier of the experiment.
        base_name (str): Name of the base series.
        experiment_params (dict): Parameters for the experiment.

    Returns:
        tuple: Residuals, linear forecasts, base information and execution config.
    """
    linear_fold, linear_title = generics.format_names(
        experiment_id,
        base_name,
        experiment_params['linear_model_name'],
    )

    exec_config = {
        "test_size": experiment_params['test_size'],
        "val_size": experiment_params['val_size'],
        "horizon": experiment_params['horizon'],
        'lag_size': experiment_params['lag_size'],
        'diff_kpss': experiment_params['diff_kpss'],
        'normalize': False,
        'type_filter': None,
    }

    base_info = input.open_format_train_val_test(base_name, exec_config)

    pn = generics.open_saved_result(linear_title)[0]['experiment'].metrics_results

    if pn['val_predict'] is not None:
        ts_forecast = np.concatenate((pn['train_predict'], pn['val_predict'], pn['test_predict']), axis=0)
    else:
        ts_forecast = np.concatenate((pn['train_predict'], pn['test_predict']), axis=0)

    if pn.get('residual_series', None) is None:
        if ts_forecast.shape[0] != base_info.original_ts.shape[0]:
            raise Exception("Size of linear model forecast must be the same size of the time series")

        error_series = np.subtract(base_info.original_ts, ts_forecast)
    else:
        error_series = pn['residual_series']

    return (
        error_series[base_info.lag_size_formated:],
        ts_forecast[base_info.lag_size_formated:],
        base_info,
        exec_config,
    )


class Additive:
    """Hybrid additive model combining linear and nonlinear forecasts."""

    def __init__(
        self,
        model: object,
        experiment_id: str,
        base_name: str,
        model_name: str,
        force: bool = True,
        normalize: bool = True,
        experiment_params: dict = None,
    ) -> None:
        """Initialize the additive hybrid model wrapper."""
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params or {}
        self.force = force
        self.normalize = normalize

    def fit_predict(self) -> None:
        """Train the nonlinear component and combine it with the linear forecast."""
        (
            error_series,
            ts_forecast,
            base_info,
            exec_config,
        ) = input_linear_info(
            self.experiment_id,
            self.base_name,
            self.experiment_params,
        )
        original_ts = base_info.original_ts
        test_size = base_info.test_size
        val_size = base_info.val_size
        lag_size = base_info.lag_size_formated
        normalize = self.normalize

        residual_result = generics.fit_predict_model(
            self.model,
            pd.Series(error_series),
            normalize,
            lag_size,
            exec_config,
            False,
        )

        all_residual_forecasts = np.concatenate((
            residual_result['train_predict'],
            residual_result['val_predict'],
            residual_result['test_predict'],
        ), axis=0)

        final_forecast = ts_forecast[lag_size:] + all_residual_forecasts

        train_predict = final_forecast[0:-(test_size + val_size)]
        val_predict = final_forecast[-(test_size + val_size):-test_size]
        test_predict = final_forecast[-test_size:]
        test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict)

        if val_size is not None and val_size > 0:
            val_metrics = metrics.gerenerate_metric_results(original_ts[-(test_size + val_size):-test_size], val_predict)
        else:
            val_metrics = None

        self.metrics_results = {
            'train_predict': train_predict,
            'val_predict': val_predict,
            'test_predict': test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': residual_result['time_exec'],
            'linear_forecast': ts_forecast[lag_size:],
            'nonlinear_forecast': all_residual_forecasts,
        }


class NonLinear:
    """Nonlinear component model built on top of residuals."""

    def __init__(
        self,
        model: object,
        experiment_id: str,
        base_name: str,
        model_name: str,
        force: bool = True,
        normalize: bool = True,
        experiment_params: dict = None,
    ) -> None:
        """Initialize the nonlinear component wrapper."""
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params or {}
        self.force = force
        self.normalize = normalize

    def format_input_output(self, ts_univariate: np.ndarray, error_series: np.ndarray, ts_forecast: np.ndarray, lag_size: int, is_stationary: bool) -> tuple:
        """Create lagged input and target dataframes for the residual model."""
        if not is_stationary:
            ts_univariate = [0] + list(ts_univariate)

        ts_wind = input.create_windowing(pd.DataFrame(ts_univariate), lag_size)
        error_wind = input.create_windowing(pd.DataFrame(error_series), lag_size)
        forecast_wind = input.create_windowing(pd.DataFrame(ts_forecast), lag_size)

        use_linear = self.experiment_params["use_linear"]
        use_error = self.experiment_params["use_error"]
        use_series = self.experiment_params["use_series"]
        df_input = pd.DataFrame()

        if use_linear:
            columns_name = [f'linear_{i}' for i in reversed(range(1, lag_size + 1))] + ['actual_linear']
            forecast_wind.columns = columns_name
            df_input = pd.concat([df_input, forecast_wind[['actual_linear']]], axis=1)

        if use_error:
            columns_name = [f'error_{i}' for i in reversed(range(1, lag_size + 1))] + ['actual']
            error_wind.columns = columns_name
            df_input = pd.concat([df_input, error_wind.drop(columns=['actual'])], axis=1)

        if use_series:
            df_input = pd.concat([df_input, ts_wind.drop(columns=['actual'])], axis=1)

        df_output = ts_wind[['actual']]

        return df_input, df_output

    def split_training_test(self, df_input: pd.DataFrame, df_output: pd.DataFrame, base_info: object) -> tuple:
        """Split the prepared dataframe into training, validation and test sets."""
        df_input_norm = df_input.copy()
        df_output_norm = df_output.copy()
        min_max_scaler_y = None
        if self.normalize:
            min_max_scaler_x = preprocessing.MinMaxScaler()
            min_max_scaler_y = preprocessing.MinMaxScaler()

            min_max_scaler_x.fit(df_input.iloc[0:-base_info.test_size:])
            min_max_scaler_y.fit(df_output.iloc[0:-base_info.test_size:])

            df_input_norm = min_max_scaler_x.transform(df_input)
            df_output_norm = min_max_scaler_y.transform(df_output)

        y_train = df_output_norm[0:-(base_info.test_size + base_info.val_size)].flatten()
        x_train = df_input_norm[0:-(base_info.test_size + base_info.val_size)]

        y_val = df_output_norm[-(base_info.test_size + base_info.val_size):-base_info.test_size].flatten()
        x_val = df_input_norm[-(base_info.test_size + base_info.val_size):-base_info.test_size]

        x_test = df_input_norm[-base_info.test_size:]

        return x_train, y_train, x_val, y_val, x_test, min_max_scaler_y

    def fit_predict(self) -> None:
        """Fit the nonlinear model and format its forecasting outputs."""
        (
            error_series,
            ts_forecast,
            base_info,
            exec_config,
        ) = input_linear_info(
            self.experiment_id,
            self.base_name,
            self.experiment_params,
        )
        test_size = base_info.test_size
        val_size = base_info.val_size
        lag_size = base_info.lag_size_formated

        df_input, df_output = self.format_input_output(
            base_info.ts_univariate,
            error_series,
            ts_forecast,
            lag_size,
            base_info.is_stationary,
        )

        (
            x_train, y_train, x_val, y_val, x_test, min_max_scaler,
        ) = self.split_training_test(df_input, df_output, base_info)

        (
            train_predict,
            val_predict,
            test_predict,
            time_exec,
        ) = generics.fit_predict_ml_schemma(self.model, x_train, y_train, x_val, x_test)

        self.metrics_results = generics.format_forecats(
            base_info.original_ts,
            time_exec,
            test_size,
            val_size,
            self.normalize,
            min_max_scaler,
            base_info.is_stationary,
            self.experiment_params['diff_kpss'],
            y_val,
            train_predict,
            val_predict,
            test_predict,
        )


class RecursiveAdditive:
    """Recursive additive hybrid model that iteratively refines residual forecasts."""

    def __init__(
        self,
        model: object,
        experiment_id: str,
        base_name: str,
        model_name: str,
        force: bool = True,
        normalize: bool = True,
        experiment_params: dict = None,
    ) -> None:
        """Initialize the recursive additive hybrid wrapper."""
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params or {}
        self.force = force
        self.normalize = normalize

    def exec_recusive_forecast(self, ts_forecast: np.ndarray, error_series: np.ndarray, base_info: object, exec_config: dict) -> tuple:
        """Run the recursive residual forecasting loop."""
        original_ts = base_info.original_ts
        lag_size = base_info.lag_size_formated
        normalize = self.normalize
        max_it = self.experiment_params['max_it']
        learning_rate = self.experiment_params['learning_rate']

        forecast_df = pd.DataFrame({'linear': ts_forecast})
        time_exec = {}
        for i in range(0, max_it):
            residual_result = generics.fit_predict_model(
                clone(self.model),
                pd.Series(error_series),
                normalize,
                lag_size,
                exec_config,
                False,
            )

            all_residual_forecasts = np.concatenate((
                residual_result['train_predict'],
                residual_result['val_predict'],
                residual_result['test_predict'],
            ), axis=0)

            rf_size = forecast_df.shape[0] - all_residual_forecasts.shape[0]
            forecast_df[f'rf_{i}'] = ([0] * rf_size) + list(all_residual_forecasts * learning_rate)

            final_forecast = forecast_df.sum(axis=1).values

            error_series = np.subtract(original_ts, final_forecast)

            time_exec['training'] = time_exec.get('training', 0) + residual_result['time_exec']['training']
            time_exec['testing'] = time_exec.get('testing', 0) + residual_result['time_exec']['testing']

        return forecast_df.sum(axis=1).values, time_exec

    def fit_predict(self) -> None:
        """Train the recursive additive model and store its metrics."""
        (
            error_series,
            ts_forecast,
            base_info,
            exec_config,
        ) = input_linear_info(
            self.experiment_id,
            self.base_name,
            self.experiment_params,
        )
        original_ts = base_info.original_ts
        test_size = base_info.test_size
        val_size = base_info.val_size

        final_forecast, time_exec = self.exec_recusive_forecast(
            ts_forecast,
            error_series,
            base_info,
            exec_config,
        )

        train_predict = final_forecast[0:-(test_size + val_size)]
        val_predict = final_forecast[-(test_size + val_size):-test_size]
        test_predict = final_forecast[-test_size:]
        test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict)
        if val_size is not None and val_size > 0:
            val_metrics = metrics.gerenerate_metric_results(original_ts[-(test_size + val_size):-test_size], val_predict)
        else:
            val_metrics = None

        self.metrics_results = {
            'train_predict': train_predict,
            'val_predict': val_predict,
            'test_predict': test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': time_exec,
        }


class HighLowAdditive:
    """Hybrid model that combines linear forecasts with residual forecasts."""

    def __init__(
        self,
        model: object,
        experiment_id: str,
        base_name: str,
        model_name: str,
        force: bool = True,
        normalize: bool = True,
        experiment_params: dict = None,
    ) -> None:
        """Initialize the high-low additive wrapper."""
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params or {}
        self.force = force
        self.normalize = normalize

    def fit_predict(self) -> None:
        """Fit the residual model and combine it with the linear forecast."""
        (
            error_series,
            ts_forecast,
            base_info,
            exec_config,
        ) = input_linear_info(
            self.experiment_id,
            self.base_name,
            self.experiment_params,
        )

        original_ts = base_info.original_ts
        test_size = base_info.test_size
        val_size = base_info.val_size
        lag_size = base_info.lag_size_formated
        normalize = self.normalize

        residual_result = generics.fit_predict_model(
            self.model,
            pd.Series(error_series),
            normalize,
            lag_size,
            exec_config,
            False,
        )

        all_residual_forecasts = np.concatenate((
            residual_result['train_predict'],
            residual_result['val_predict'],
            residual_result['test_predict'],
        ), axis=0)

        final_forecast = ts_forecast[lag_size:] + all_residual_forecasts

        train_predict = final_forecast[0:-(test_size + val_size)]
        val_predict = final_forecast[-(test_size + val_size):-test_size]
        test_predict = final_forecast[-test_size:]
        test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict)

        if val_size is not None and val_size > 0:
            val_metrics = metrics.gerenerate_metric_results(original_ts[-(test_size + val_size):-test_size], val_predict)
        else:
            val_metrics = None

        self.metrics_results = {
            'train_predict': train_predict,
            'val_predict': val_predict,
            'test_predict': test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': residual_result['time_exec'],
        }


class ResidualCombination:
    """Ensemble-like wrapper for combining linear and residual forecasts."""

    def __init__(
        self,
        model: object,
        experiment_id: str,
        base_name: str,
        model_name: str,
        force: bool = True,
        normalize: bool = True,
        experiment_params: dict = None,
    ) -> None:
        """Initialize the residual combination wrapper."""
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params or {}
        self.force = force
        self.normalize = normalize

    def split_training_test(self, df_input: pd.DataFrame, df_output: pd.DataFrame, base_info: object) -> tuple:
        """Split a combined forecast dataframe into train/validation/test sets."""
        df_input_norm = df_input.copy()
        df_output_norm = df_output.copy()
        min_max_scaler_y = None

        if self.normalize:
            min_max_scaler_x = preprocessing.MinMaxScaler()
            min_max_scaler_y = preprocessing.MinMaxScaler()

            min_max_scaler_x.fit(df_input.iloc[0:-base_info.test_size:])
            min_max_scaler_y.fit(df_output.iloc[0:-base_info.test_size:])

            df_input_norm = min_max_scaler_x.transform(df_input)
            df_output_norm = min_max_scaler_y.transform(df_output)

        y_train = df_output_norm[0:-(base_info.test_size + base_info.val_size)].flatten()
        x_train = df_input_norm[0:-(base_info.test_size + base_info.val_size)]

        y_val = df_output_norm[-(base_info.test_size + base_info.val_size):-base_info.test_size].flatten()
        x_val = df_input_norm[-(base_info.test_size + base_info.val_size):-base_info.test_size]

        x_test = df_input_norm[-base_info.test_size:]

        return x_train, y_train, x_val, y_val, x_test, min_max_scaler_y

    def exec_comb(self) -> None:
        """Combine linear and nonlinear forecasts and save the result."""
        lag_size_base = self.experiment_params['lag_size']
        diff_kpss = self.experiment_params['diff_kpss']
        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size'],
            'horizon': self.experiment_params['horizon'],
        }

        nonlinear_fold, nonlinear_title = generics.format_names(
            self.experiment_id,
            self.base_name,
            self.experiment_params['nonlinear_model'],
        )

        fold, title = generics.format_names(
            self.experiment_id,
            self.base_name,
            self.model_name,
        )

        list_execs = generics.open_saved_result(nonlinear_title)

        base_info = input.open_format_train_val_test(
            self.base_name,
            False,
            lag_size_base,
            exec_config,
            diff_kpss,
        )
        predict_results = []
        for le in list_execs:
            ts_size = le['experiment'].metrics_results['nonlinear_forecast'].shape[0]
            df_input = pd.DataFrame({
                'linear_forecast': le['experiment'].metrics_results['linear_forecast'],
                'nonlinear_forecast': le['experiment'].metrics_results['nonlinear_forecast'],
            })

            df_input['ens'] = df_input['linear_forecast'] + df_input['nonlinear_forecast']
            df_output = pd.DataFrame({'actual': base_info.ts_univariate[-ts_size:].copy()})

            x_train, y_train, x_val, y_val, x_test, min_max_scaler_y = self.split_training_test(df_input, df_output, base_info)
            (
                train_predict,
                val_predict,
                test_predict,
                time_exec,
            ) = generics.fit_predict_ml_schemma(self.model, x_train, y_train, x_val, x_test)

            test_size = base_info.test_size
            val_size = base_info.val_size
            metric_result = generics.format_forecats(
                base_info.original_ts,
                time_exec,
                test_size,
                val_size,
                self.normalize,
                min_max_scaler_y,
                base_info.is_stationary,
                self.experiment_params['diff_kpss'],
                y_val,
                train_predict,
                val_predict,
                test_predict,
            )

            model_exp_test = ResultExp(metric_result)

            predict_results.append({'experiment': model_exp_test, 'val_metric': np.inf})

        generics.save_result(fold, title, predict_results)


class AdditiveValidFits:
    """Hybrid model that uses valid-fit residuals from a linear model."""

    def __init__(
        self,
        model: object,
        experiment_id: str,
        base_name: str,
        model_name: str,
        force: bool = True,
        normalize: bool = True,
        experiment_params: dict = None,
    ) -> None:
        """Initialize the additive valid-fits wrapper."""
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params or {}
        self.force = force
        self.normalize = normalize

    def fit_predict(self) -> None:
        """Use residual series from valid linear fits and combine them with the linear forecast."""
        linear_fold, linear_title = generics.format_names(
            self.experiment_id,
            self.base_name,
            self.experiment_params['linear_model_name'],
        )

        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size'],
            "horizon": self.experiment_params['horizon'],
            'lag_size': self.experiment_params['lag_size'],
            'diff_kpss': self.experiment_params['diff_kpss'],
            'normalize': False,
            'type_filter': None,
        }

        base_info = input.open_format_train_val_test(self.base_name, exec_config)
        pn = generics.open_saved_result(linear_title)[0]['experiment']
        test_size = base_info.test_size
        val_size = base_info.val_size
        original_ts = base_info.original_ts

        df_train = pd.DataFrame()
        df_val = pd.DataFrame()
        df_test = pd.DataFrame()

        for i, c in enumerate(pn.error_series.columns):
            ts_actual = pn.error_series[c]
            df_windowed = input.create_windowing(
                pd.DataFrame({'y': ts_actual.values}),
                base_info.lag_size_formated + (exec_config['horizon'] - 1),
            )
            df_train = pd.concat([df_train, df_windowed.iloc[0:-(test_size + val_size)]], axis=0)
            df_val = pd.concat([df_val, df_windowed.iloc[-(test_size + val_size):-test_size]], axis=0)
            if i == 0:  # only best arima
                df_test = df_windowed.copy()

        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        insample_data = pd.concat([df_train, df_val])

        min_max_scaler_x = preprocessing.MinMaxScaler()
        min_max_scaler_x.fit(insample_data.drop(columns='actual'))

        min_max_scaler_y = preprocessing.MinMaxScaler()
        min_max_scaler_y.fit(insample_data['actual'].values.reshape(-1, 1))

        y_train = min_max_scaler_y.transform(df_train['actual'].values.reshape(-1, 1))
        x_train = min_max_scaler_x.transform(df_train.drop(columns=['actual']))

        x_val = min_max_scaler_x.transform(df_val.drop(columns=['actual']))

        x_test = min_max_scaler_x.transform(df_test.drop(columns=['actual']))
        (
            train_predict,
            val_predict,
            test_predict,
            time_exec,
        ) = generics.fit_predict_ml_schemma(self.model, x_train, y_train, x_val, x_test)

        all_residual_forecasts = min_max_scaler_y.inverse_transform(test_predict.reshape(-1, 1)).flatten()

        linear_forecast = pn.df_prevs[pn.df_prevs.columns[0]]
        final_forecast = linear_forecast[base_info.lag_size_formated:] + all_residual_forecasts

        test_predict = final_forecast[-test_size:]
        train_predict = final_forecast[0:-(test_size + val_size)]
        if x_val.shape[0] > 0:
            val_predict = final_forecast[-(test_size + val_size):-test_size]

        test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict)

        if val_size is not None and val_size > 0:
            val_metrics = metrics.gerenerate_metric_results(original_ts[-(test_size + val_size):-test_size], val_predict)
        else:
            val_metrics = None

        self.metrics_results = {
            'train_predict': train_predict,
            'val_predict': val_predict,
            'test_predict': test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': time_exec,
            'linear_forecast': df_test['actual'].values[base_info.lag_size_formated:],
            'nonlinear_forecast': all_residual_forecasts,
        }