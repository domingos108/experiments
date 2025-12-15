import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.base import clone

from input import input
from model import generics
from metrics import metrics

def invscaling(count, power_t, learning_rate_init):
    return learning_rate_init / np.power(count, power_t)

def input_linear_info(experiment_id, base_name, experiment_params, model_name):
    linear_fold, linear_title = generics.format_names(
        experiment_id, 
        base_name, 
        experiment_params['linear_model_name']
    )
    lag_size_base = experiment_params['lag_size']
    diff_kpss = experiment_params['diff_kpss']

    exec_config = {
        "test_size": experiment_params['test_size'],
        "val_size": experiment_params['val_size']
    }

    base_info = input.open_format_train_val_test(
        base_name, 
        False, 
        lag_size_base, 
        exec_config, 
        diff_kpss
    )
   
    pn = generics.open_saved_result(
        linear_title
    )[0]['experiment'].metrics_results
    
    ts_forecast = np.concatenate((pn['train_predict'], pn['test_predict']), axis=0)

    fold, title = generics.format_names(experiment_id, base_name, model_name)

    if ts_forecast.shape[0] != base_info.original_ts.shape[0]:
        raise Exception("Size of linear model forecast must be the same size of the time series")
    
    error_series = np.subtract(base_info.original_ts, ts_forecast)

    return (
        error_series,
        ts_forecast,
        base_info,
        exec_config
    )

class Additive:
    def __init__(self, 
                 model,
                 experiment_id, 
                 base_name, 
                 model_name, 
                 force=True,
                 normalize = True,
                 experiment_params = {} 

        ):
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params
        self.force = force
        self.normalize = normalize

    def fit_predict(self):

        (
            error_series,
            ts_forecast,
            base_info,
            exec_config
        ) = input_linear_info(
            self.experiment_id, 
            self.base_name, 
            self.experiment_params, 
            self.model_name, 
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
            False
        )

        all_residual_forecasts = np.concatenate((
            residual_result['train_predict'], 
            residual_result['val_predict'], 
            residual_result['test_predict'] 
        ), axis=0)

        final_forecast = ts_forecast[lag_size:] + all_residual_forecasts 

        train_predict = final_forecast[0:-(test_size+val_size)]
        val_predict = final_forecast[-(test_size+val_size): -test_size]
        test_predict = final_forecast[-test_size:]
        test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict)
        if val_size!=None and val_size>0:
            val_metrics = metrics.gerenerate_metric_results(original_ts[-(test_size+val_size): -test_size], val_predict)
        else:
            val_metrics = None

        self.metrics_results = {
            'train_predict': train_predict, 
            'val_predict': val_predict, 
            'test_predict':test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': residual_result['time_exec']
        }


class NonLinear:
    def __init__(self, 
                 model,
                 experiment_id, 
                 base_name, 
                 model_name, 
                 force=True,
                 normalize = True,
                 experiment_params = {} 

        ):

        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params
        self.force = force
        self.normalize = normalize
    
    def format_input_output(self, ts_univariate, error_series, ts_forecast, lag_size, is_stationary):
        if not is_stationary:
            ts_univariate = [0]+list(ts_univariate)

        ts_wind = input.create_windowing(pd.DataFrame(ts_univariate), lag_size)
        error_wind = input.create_windowing(pd.DataFrame(error_series), lag_size)
        forecast_wind = input.create_windowing(pd.DataFrame(ts_forecast), lag_size)

        use_linear = self.experiment_params["use_linear"]
        use_error = self.experiment_params["use_error"]
        use_series = self.experiment_params["use_series"]
        df_input = pd.DataFrame()
        if use_linear:
            columns_name = [f'linear_{i}' for i in reversed(range(1, lag_size+1))] + ['actual_linear']
            forecast_wind.columns = columns_name
            df_input = pd.concat([df_input, forecast_wind[['actual_linear']]], axis=1)

        if use_error:
            columns_name = [f'error_{i}' for i in reversed(range(1, lag_size+1))] + ['actual']
            error_wind.columns = columns_name
            df_input = pd.concat([df_input, error_wind.drop(columns=['actual'])], axis=1)

        if use_series:
            df_input = pd.concat([df_input, ts_wind.drop(columns=['actual'])], axis=1)

        df_output = ts_wind[['actual']]

        return df_input, df_output

    def split_training_test(self, df_input, df_output, base_info):
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

        y_train = df_output_norm[0:-(base_info.test_size+base_info.val_size)].flatten()
        x_train = df_input_norm[0:-(base_info.test_size+base_info.val_size)]

        y_val = df_output_norm[-(base_info.test_size+base_info.val_size): -base_info.test_size].flatten()
        x_val = df_input_norm[-(base_info.test_size+base_info.val_size): -base_info.test_size]

        x_test = df_input_norm[-base_info.test_size:]

        return x_train, y_train, x_val, y_val, x_test, min_max_scaler_y

    def fit_predict(self):
        (
            error_series,
            ts_forecast,
            base_info,
            exec_config
        ) = input_linear_info(
            self.experiment_id, 
            self.base_name, 
            self.experiment_params, 
            self.model_name, 
        )
        test_size = base_info.test_size
        val_size = base_info.val_size
        lag_size = base_info.lag_size_formated
        
        df_input, df_output = self.format_input_output(
            base_info.ts_univariate, 
            error_series, 
            ts_forecast, 
            lag_size,
            base_info.is_stationary
        )

        (
           x_train, y_train, x_val, y_val, x_test, min_max_scaler
        ) =  self.split_training_test(df_input, df_output, base_info)

        (
            train_predict, 
            val_predict, 
            test_predict,
            time_exec
        ) = generics.fit_predict_ml_schemma(self.model, x_train, y_train, x_val, x_test)


        self.metrics_results = generics.format_forecats(base_info.original_ts, 
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
                        test_predict

        )


class RecursiveAdditive:
    def __init__(self, 
                 model,
                 experiment_id, 
                 base_name, 
                 model_name, 
                 force=True,
                 normalize = True,
                 experiment_params = {} 

        ):
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params
        self.force = force
        self.normalize = normalize

    def exec_recusive_forecast(self, ts_forecast, error_series, base_info, exec_config):

        original_ts = base_info.original_ts
        lag_size = base_info.lag_size_formated
        normalize = self.normalize
        max_it = self.experiment_params['max_it']
        learning_rate = self.experiment_params['learning_rate']
                                               
        forecast_df = pd.DataFrame({
            'linear': ts_forecast
        })
        time_exec = {}
        for i in range(0, max_it):        
            residual_result = generics.fit_predict_model(
                clone(self.model), 
                pd.Series(error_series), 
                normalize, 
                lag_size, 
                exec_config, 
                False
            )

            all_residual_forecasts = np.concatenate((
                residual_result['train_predict'], 
                residual_result['val_predict'], 
                residual_result['test_predict'] 
            ), axis=0)

            rf_size = ( forecast_df.shape[0] - all_residual_forecasts.shape[0])
            #lr_actual =  invscaling(max_it  - i, 0.9, learning_rate)
            forecast_df[f'rf_{i}'] = ([0] * rf_size) + list(all_residual_forecasts * learning_rate)

            final_forecast = forecast_df.sum(axis=1).values

            error_series = np.subtract(original_ts, final_forecast)

            time_exec['training'] = (
                time_exec.get('training', 0 ) + residual_result['time_exec']['training']
            )

            time_exec['testing'] = (
                time_exec.get('testing', 0 ) + residual_result['time_exec']['testing']
            )

        return forecast_df.sum(axis=1).values, time_exec

    def fit_predict(self):

        (
            error_series,
            ts_forecast,
            base_info,
            exec_config
        ) = input_linear_info(
            self.experiment_id, 
            self.base_name, 
            self.experiment_params, 
            self.model_name, 
        )
        original_ts = base_info.original_ts
        test_size = base_info.test_size
        val_size = base_info.val_size

        final_forecast, time_exec = self.exec_recusive_forecast(
            ts_forecast, 
            error_series, 
            base_info, 
            exec_config
        )
       
        train_predict = final_forecast[0:-(test_size+val_size)]
        val_predict = final_forecast[-(test_size+val_size): -test_size]
        test_predict = final_forecast[-test_size:]
        test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict)
        if val_size!=None and val_size>0:
            val_metrics = metrics.gerenerate_metric_results(original_ts[-(test_size+val_size): -test_size], val_predict)
        else:
            val_metrics = None

        self.metrics_results = {
            'train_predict': train_predict, 
            'val_predict': val_predict, 
            'test_predict':test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': time_exec
        }

