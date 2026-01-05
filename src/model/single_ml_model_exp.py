import time

import numpy as np

from input import input
from metrics import metrics

def inverse_difference(last_ob, value):
    return value + last_ob

def get_inverse_difference(last_ob,value):
     return [inverse_difference(last_ob[i], value[i]) for i in range(len(value))]

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

class SKlearnModel:
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
        lag_size_base = self.experiment_params['lag_size']
        horizon = self.experiment_params.get('horizon', 1)

        normalize = self.normalize
        diff_kpss = self.experiment_params.get('diff_kpss', True)

        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size'],
            'horizon': horizon
        }
    
        base_info = input.open_format_train_val_test(
            self.base_name, normalize, lag_size_base, exec_config, diff_kpss)
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

        (
            train_predict, 
            val_predict, 
            test_predict,
            time_exec
        ) = fit_predict_ml_schemma(self.model, x_train, y_train, x_val, x_test)
        
        y_train_original = original_ts[0:-(test_size+val_size)][- len(train_predict):]
        y_test_original = original_ts[-test_size:]
        y_val_original = original_ts[-(test_size+val_size): -test_size]

        if normalize:
            test_predict = min_max_scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
            train_predict = min_max_scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
            if y_val.shape[0] > 0:
                val_predict = min_max_scaler.inverse_transform(val_predict.reshape(-1, 1)).flatten()
            
        if (is_stationary is False) and (diff_kpss is True):
            train_predict = get_inverse_difference(original_ts[0:-(test_size+val_size+horizon)], train_predict)
            
            if y_val.shape[0] > 0:
        
                val_predict =  get_inverse_difference(original_ts[-(test_size+val_size+horizon): -(test_size+horizon)], val_predict)
            
            test_predict =  get_inverse_difference(original_ts[-(test_size+horizon):-horizon], test_predict)

        test_metrics = metrics.gerenerate_metric_results(y_test_original, test_predict)

        if y_val.shape[0]==0:
            val_metrics={}
        else:
            val_metrics = metrics.gerenerate_metric_results(y_val_original, val_predict)

        self.metrics_results = {
            'train_predict': train_predict, 
            'val_predict': val_predict, 
            'test_predict':test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': time_exec
        }