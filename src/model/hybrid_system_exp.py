from input import input
from model import generics
from metrics import metrics
import numpy as np

import pandas as pd

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
        linear_fold, linear_title = generics.format_names(
            self.experiment_id, 
            self.base_name, 
            self.experiment_params['linear_model_name']
        )
        lag_size = self.experiment_params['lag_size']
        normalize = self.normalize
        diff_kpss = False

        pn = generics.open_saved_result(linear_title)[0]
        ts_forecast = np.concatenate((pn['train_predict'], pn['test_predict']), axis=0)

        fold, title = generics.format_names(self.experiment_id, self.base_name, self.model_name)

        if generics.file_exists(title) and (not self.force):
            print('Modelo já executado')       

        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size']
        }

        (
            _,
            _, 
            _,
            _, 
            _,
            test_size, 
            val_size,
            _, 
            original_ts
        ) = input.open_format_train_val_test(self.base_name, normalize, lag_size, exec_config, diff_kpss)

        if ts_forecast.shape[0] != original_ts.shape[0]:
            raise Exception("Size of linear model forecast must be the same size of the time series")
        
        error_series = np.subtract(original_ts, ts_forecast)

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

