import time
import os
import logging
import warnings

import torch
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast

from input import input
from metrics import metrics


os.environ['PYTORCH_LIGHTNING_LOG_LEVEL'] = 'ERROR'
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="`isinstance\(treespec, LeafSpec\)` is deprecated")


# Força o PyTorch a usar todos os núcleos lógicos
num_cores = os.cpu_count()
torch.set_num_threads(num_cores)

class NeuralForecastExp:
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

    def predict_steps(self, df_prev, fcst, model_name_fcst, current_train, shape_prevs):
        test_prevs = []
        start_time_test = time.time()

        for i in range(df_prev.shape[0]):
            # Prevemos o próximo ponto
            # O Nixtla usa o final do DataFrame fornecido para gerar a previsão
            forecast = fcst.predict(df=current_train)
            test_prevs.append(forecast[model_name_fcst].iloc[self.experiment_params['horizon'] -1])
            
            # "Update": Adicionamos a linha real do teste ao DataFrame de treino
            actual_row =  df_prev[['unique_id', 'ds', 'y']].iloc[[i]]
            current_train = pd.concat([current_train, actual_row]).reset_index(drop=True)

            # for multiple step forecasts
            if shape_prevs == len(test_prevs):
                break

        end_time_test = time.time()

        return test_prevs, end_time_test, start_time_test 

    
    def fit_predict(self):
 
        lag_size_base = self.experiment_params['lag_size']
        normalize = self.normalize
        diff_kpss = self.experiment_params.get('diff_kpss', True)

        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size'],
            "horizon": self.experiment_params['horizon'],
            'normalize': normalize,
            'diff_kpss': diff_kpss,
            'lag_size': lag_size_base,
            'type_filter': None
        }
    
        base_info = input.open_format_train_val_test(
            self.base_name, exec_config
        )
        
        params = self.experiment_params['model_actual_config']
        params['h'] = self.experiment_params['horizon']
        params['random_seed'] = np.random.randint(0, 10000)
        params['input_size'] = base_info.lag_size_formated
        params['logger'] = False
        params['enable_progress_bar']= False
        params['enable_model_summary']= False
        params['default_root_dir'] = None
    
        model_base = self.model(**params)

        ts_univariate = base_info.ts_univariate
        test_size = base_info.test_size
        val_size = base_info.val_size
        freq = base_info.freq

        df_ts = pd.DataFrame(
            {
                'unique_id': '1',
                'ds': pd.date_range(start='1900-01-01', periods=ts_univariate.shape[0], freq = freq),
                'y': ts_univariate
            }
        )

        df_training = df_ts.iloc[0: - (test_size + val_size+ self.experiment_params['horizon']-1)]

        df_prev = df_ts.iloc[- (test_size + val_size + self.experiment_params['horizon']-1):]

        start_time_training = time.time()
        fcst = NeuralForecast(models=[model_base], freq=freq)
        fcst.fit(df=df_training)
        end_time_training = time.time()

        model_name_fcst = [model.__class__.__name__ for model in fcst.models]

        if len(model_name_fcst) > 1:
            raise NotImplementedError("Multiplos modelos não implementado")
        
        model_name_fcst = model_name_fcst[0]
        train_predict = fcst.predict_insample(step_size=1)[model_name_fcst]

        current_train = df_training[['unique_id', 'ds', 'y']].copy()

        shape_prevs = test_size + val_size
        test_prevs, end_time_test, start_time_test = self.predict_steps(
            df_prev, fcst, model_name_fcst, current_train, shape_prevs
        )
        
        if val_size>0:
            val_predict = test_prevs[0:-test_size]
        else:
            val_predict = np.array([])
        
        test_predict = test_prevs[-test_size:]

        y_train_original = base_info.original_ts[0:-(test_size+val_size)][- len(train_predict):]
        y_test_original = base_info.original_ts[-test_size:]
        y_val_original = base_info.original_ts[-(test_size+val_size): -test_size]

        if (base_info.is_stationary is False) and (diff_kpss is True):
            train_predict =  y_train_original + train_predict
            
            if val_size > 0:
                test_predict =  np.concatenate(( [y_val_original[-1]], y_test_original[0:-1])) + test_predict

                val_predict =  np.concatenate(( [y_train_original[-1]], y_val_original[0:-1])) + val_predict
            else:
                test_predict =  np.concatenate(( [y_train_original[-1]], y_test_original[0:-1])) + test_predict

        test_metrics = metrics.gerenerate_metric_results(y_test_original, test_predict)

        if val_size > 0:
            val_metrics = metrics.gerenerate_metric_results(y_val_original, val_predict)
        else:
            val_metrics={}

        self.metrics_results = {
            'train_predict': train_predict, 
            'val_predict': val_predict, 
            'test_predict':test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': {
                'testing': end_time_test - start_time_test, 
                'training': end_time_training - start_time_training
            }
        }
        