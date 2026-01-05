import os
import logging
import copy

from sklearn import preprocessing
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast

from metrics import metrics
from input import input


os.environ['PYTORCH_LIGHTNING_LOG_LEVEL'] = 'ERROR'
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def create_min_max_scale(ts):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(ts.reshape(-1, 1))

    ts_norm = min_max_scaler.transform(ts.reshape(-1, 1)).flatten()

    return ts_norm, min_max_scaler


def calculate_error_and_new_ts(df_pert):
    df_p_final = df_pert.dropna().drop(columns=['actual'])
  

    y_actual = df_pert.dropna()['actual'].values
    p_final= df_p_final.sum(axis=1).values
    
    error = y_actual - p_final

    absolut_error = np.mean(abs(error))

    return error, absolut_error

class Perturbative:
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

        self.min_max_scaler = []
        self.learning_rate_init = experiment_params['learning_rate_init']
        self.power_t = 0.9
        self.fixed_error_lags = False
        self.qtd_perturbations = experiment_params['qtd_perturbations']

    def create_model_forecast(self, df_training, df_prev):

        ts_norm, min_max_scaler = create_min_max_scale(df_training['y'].values)
        df_training['y'] = ts_norm
        df_prev['y'] = min_max_scaler.transform(df_prev['y'].values.reshape(-1, 1)).flatten()

        model_base = copy.deepcopy(self.model)(**self.params)
        fcst = NeuralForecast(models=[model_base], freq=self.base_info.freq)
        fcst.fit(df=df_training)
        
        model_name = [model.__class__.__name__ for model in fcst.models]

        if len(model_name) > 1:
            raise NotImplementedError("Multiplos modelos não implementado")
        
        model_name = model_name[0]
        train_predict = fcst.predict_insample(step_size=1)[
            model_name
        ][self.base_info.lag_size_formated:]

        train_predict = min_max_scaler.inverse_transform(train_predict.values.reshape(-1, 1)).flatten()
        train_predict = train_predict.tolist()

        current_train = df_training[['unique_id', 'ds', 'y']].copy()
        test_prevs = []

        for i in range(df_prev.shape[0]):
            # Prevemos o próximo ponto
            # O Nixtla usa o final do DataFrame fornecido para gerar a previsão
            forecast = fcst.predict(df=current_train)
            test_prevs.append(forecast[model_name].iloc[self.params['h'] -1])
            
            # "Update": Adicionamos a linha real do teste ao DataFrame de treino
            actual_row =  df_prev[['unique_id', 'ds', 'y']].iloc[[i]]
            current_train = pd.concat([current_train, actual_row]).reset_index(drop=True)

        test_prevs = np.array(test_prevs)
        test_prevs = min_max_scaler.inverse_transform(test_prevs.reshape(-1, 1)).flatten().tolist()

        return  train_predict + test_prevs


    def perturb(self, df_ts: pd.DataFrame):

        if self.base_info.original_ts.ndim > 1:
            raise Exception('y must be a univariate numpy array')
        
        p_components = df_ts.copy()

        test_size = self.base_info.test_size
        val_size = self.base_info.val_size
        
        df_pert = pd.DataFrame(
           { 'actual': df_ts['y'].values}
        )

        for ip in range(0, self.qtd_perturbations+1):
            df_training = p_components.iloc[0: - (test_size + val_size)].copy()
            df_prev = p_components.iloc[- (test_size + val_size):].copy()

            p_prev = self.create_model_forecast(df_training, df_prev)

            p_i_size = ( df_pert.shape[0] - len(p_prev))

            df_pert[f'p{ip}'] = ([None] * p_i_size) + p_prev

            error, absolut_error = calculate_error_and_new_ts(df_pert)
            
            p_components = pd.DataFrame(
                {
                    'unique_id': '1',
                    'ds': pd.date_range(start='1900-01-01', periods=error.shape[0], freq = self.base_info.freq),
                    'y': error
                }
            )

        final_forecast = df_pert.dropna().drop(
            columns=['actual']
        ).sum(axis=1).values
            
        return final_forecast, df_pert

    def fit_predict(self):
       
        lag_size_base = self.experiment_params['lag_size']
        normalize = self.normalize
        diff_kpss = self.experiment_params.get('diff_kpss', False)

        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size']
        }
    
        self.base_info = input.open_format_train_val_test(
            self.base_name, normalize, lag_size_base, exec_config, diff_kpss)
        
        self.params = self.experiment_params['model_actual_config']
        self.params['h'] = 1
        self.params['random_seed'] = np.random.randint(0, 10000)
        self.params['logger'] = False
        self.params['enable_progress_bar']= False
        self.params['enable_model_summary']= False
        self.params['default_root_dir'] = None
        self.params['input_size'] = self.base_info.lag_size_formated
      
        ts_univariate = self.base_info.ts_univariate
        
        freq = self.base_info.freq

        df_ts = pd.DataFrame(
            {
                'unique_id': '1',
                'ds': pd.date_range(start='1900-01-01', periods=ts_univariate.shape[0], freq = freq),
                'y': ts_univariate
            }
        )

        final_forecast, self.df_pert = self.perturb(df_ts)

        self.metrics_results = metrics.format_metrics_results(final_forecast, self.base_info)