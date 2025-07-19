
from input import input
import config
from metrics import metrics
from model import generics

from sklearn.base import BaseEstimator
from sklearn.base import clone

import pmdarima as pm

import warnings

# This will suppress ALL FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

class Arima(BaseEstimator):

    def __init__(self, seazonal_lag):
        self.seazonal_lag = seazonal_lag
        self.is_ts_mode = True

    def fit(self, ts):
        self.forecaster = pm.auto_arima(ts,
            start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5,
            start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2,
            max_order=5, m=self.seazonal_lag, stepwise=True, trace=True, maxiter=50
        ) 

        self.forecaster.fit(ts) 


    def predict_steps(self, ts_test):
        horizon=1
        train_predicted = self.forecaster.predict_in_sample()

        prevs_h_steps = []
        for t in ts_test:
            prevs_h_steps.append(self.forecaster.predict(horizon)[horizon-1])
            self.forecaster.update(t)

        return train_predicted, prevs_h_steps
    
class ResultExp:
    def __init__(self, metrics_results):
        self.metrics_results = metrics_results

def exec_training_testing(base_name, experiment_id, model_name, seazonal_lag, force=True):

    fold, title = generics.format_names(experiment_id, base_name, model_name)

    if generics.file_exists(title) and (not force):
        print('Modelo já executado')
        return None
    
    normalize = False
    diff_kpss = False
    lag_size = None
    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": 0
    }
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
    ) = base_info.sequential_return()

    model = Arima(seazonal_lag)
    forecaster = clone(model).set_params(** {})
    forecaster.fit(original_ts[0:-test_size])

    train_predict, test_predict = forecaster.predict_steps(original_ts[-test_size:])

    test_metrics = metrics.gerenerate_metric_results(original_ts[-test_size:], test_predict)
    time_exec = {'testing': None, 'training': None}
 
    result_dict = {
        'train_predict': train_predict, 
        'val_predict': None, 
        'test_predict':test_predict,
        'val_metrics': None,
        'test_metrics': test_metrics,
        'time_exec': time_exec,
        'params': None,
        'best_metric': None
    }
    result = ResultExp(result_dict)
    generics.save_result(fold, title, [{'experiment': result, 'val_metric': None}])