from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn import preprocessing
import pandas as pd
import numpy as np

from input import input
from model import generics


def create_min_max_scale(ts):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(ts.reshape(-1, 1))

    ts_norm = min_max_scaler.transform(ts.reshape(-1, 1)).flatten()

    return ts_norm, min_max_scaler

def create_window(ts, lag_size):
    df = pd.DataFrame({'y': ts})
    df_wind = input.create_windowing(df, lag_size)

    y = df_wind['actual'].values
    X = df_wind.drop(columns=['actual']).values

    return X, y 

class Perturbative(BaseEstimator):

    def __init__(
            self, 
            model_base,
            qtd_pertubacao = None,
            model_params = [],
            learning_rate_init=1,
            learning_rate = 'constant',#invscaling  
            power_t = 0.9,
            fixed_error_lags= True
        ):

        self.model_base = model_base
        self.model_params = model_params
        self.qtd_pertubacao = qtd_pertubacao
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.fixed_error_lags = fixed_error_lags

        if len(model_params) >1 and qtd_pertubacao is not None:
            raise Exception("model_params maior que 1 and qtd_pertubacao não nula ")
                
        self.model_list = []
        self.min_max_scaler = []
        self.is_ts_mode = True

        self.total_pertubs = 0

        super().__init__()

    def calculate_error_and_new_ts(self, p_components):
        
        y_actual = p_components.dropna()['actual'].values
        p_final= p_components.dropna().drop(columns=['actual']).sum(axis=1).values
        error = y_actual - p_final
    
        return error
    
    def fit(self, ts):
        if len(self.model_params) >1 and self.qtd_pertubacao is not None:
            raise Exception("model_params maior que 1 and qtd_pertubacao não nula ")
        
        if self.qtd_pertubacao is not None:
            self.model_params = self.model_params * self.qtd_pertubacao

        if ts.ndim > 1:
            raise Exception('y must be a univariate numpy array')
        
        self.total_pertubs = len(self.model_params)
        p_components= pd.DataFrame({'actual': ts})
        count = 0
        ts_actual = ts.copy()

        for params in self.model_params:
            params_actual = params.copy()

            del params_actual['time_window']
        
            lag_size_actual = params['time_window']

            (
                ts_norm, 
                min_max_scaler
            ) = create_min_max_scale(ts_actual.copy())
            
            X, y = create_window(ts_norm, lag_size_actual)
            model = clone(self.model_base).set_params(**params_actual)
            
            model = model.fit(X, y)
            
            ts_actual, p_components = self.perturbation(X, model, p_components, min_max_scaler, count)

            self.model_list.append(model)
            self.min_max_scaler.append(min_max_scaler)
            count = count + 1 
        self.train_p_components = p_components
        
        return self
    
    def invscaling(self, count):
        return self.learning_rate_init / np.power(count, self.power_t)
    
    def perturbation(self, X, model, p_components, min_max_scaler, count):
        p_i_norm = model.predict(X)
        p_i = min_max_scaler.inverse_transform(p_i_norm.reshape(-1, 1)).flatten()

        if count>0:
            learning_rate_init =  self.learning_rate_init

            if self.learning_rate == 'invscaling':
                learning_rate_init = self.invscaling(self.total_pertubs - count+1)
        else:
            learning_rate_init = 1

        p_i = p_i * learning_rate_init
        p_i_size = ( p_components.shape[0] - p_i.shape[0])
        
        if (p_components.shape[1] > 1 ) and (self.fixed_error_lags is True):
            p_i = ([0] * p_i_size) + p_i.tolist()
        else:
            p_i = ([None] * p_i_size) + p_i.tolist()
        
        p_components = pd.concat([
            p_components, 
            pd.DataFrame({f'pert{count}': p_i})
            ], axis=1)
        
        ts_actual = self.calculate_error_and_new_ts(p_components)
        
        return ts_actual, p_components

    def predict_steps(self, ts):
        """ pressupoe contexto de ts e 1 passo a frente"""
        trained_models = zip(
            self.model_list, 
            self.min_max_scaler, 
            self.model_params
        )
        ts_actual = ts.copy()
        count = 0
        
        p_components= pd.DataFrame({'actual': ts})

        for model, min_max_scaler, params in trained_models:            
        
            lag_size = params['time_window']

            ts_norm = min_max_scaler.transform(ts_actual.reshape(-1, 1)).flatten()
            X, _ = create_window(ts_norm, lag_size)

            ts_actual, p_components = self.perturbation(X, model, p_components, min_max_scaler, count)

            count = count + 1 

        self.predict_steps_p_components = p_components

        return p_components.dropna().drop(columns=['actual']).sum(axis=1).values
        

def exec_model(
        base_name_list, 
        experiment_id, 
        models_before,
        model_base, 
        params_modelo_base,
        normalize,
        model_exec,
        model_name,
        force=True
        ):

    for base_name in base_name_list:

        _, title = generics.format_names(
            experiment_id, 
            base_name, 
            models_before
        )

        pn = generics.open_saved_result(title)[0]['params']

        fixed = pn.get('model_params', [])
        if len(fixed) ==0:
            fixed = [pn]

        model = Perturbative(model_base = model_base)

        parameters = {
            'model_params': [ 
                fixed + [pmb] for pmb in params_modelo_base
            ],
            'model_base': [model_base],
            'fixed_error_lags': [False],
        }
        print(base_name)
        generics.grid_seach(
            model,  
            base_name, 
            normalize, 
            parameters, 
            model_exec,
            model_name, 
            experiment_id,
            group_metrics_name = 'val_metrics', 
            metric = 'RMSE', 
            force = force
        )