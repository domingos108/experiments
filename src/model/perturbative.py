from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn import preprocessing

import pandas as pd

from input import input

def create_min_max_scale(ts):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(ts.reshape(-1, 1))
    ts_norm = min_max_scaler.transform(ts.reshape(-1, 1)).flatten()
    return ts, None#ts_norm, min_max_scaler

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
            model_params = []
        ):
        self.model_base = model_base
        self.model_params = model_params
        
        self.model_list = []
        self.min_max_scaler = []
        self.lag_size = None
        self.is_ts_mode = True

        super().__init__()

    def calculate_error_and_new_ts(self, p_components):

        y_actual = p_components.dropna()['actual'].values
        p_final= p_components.dropna().drop(columns=['actual']).sum(axis=1).values
        error = y_actual - p_final
    
        return error
    
    def fit(self, ts, lag_size):
        self.lag_size = lag_size

        if ts.ndim > 1:
            raise Exception('y must be a univariate numpy array')
        
        p_components= pd.DataFrame({'actual': ts})
        count = 0
        ts_actual = ts.copy()
        for params_actual in self.model_params:
            (
                ts_norm, 
                min_max_scaler
            ) = create_min_max_scale(ts_actual.copy())
            
            X, y = create_window(ts_norm, self.lag_size)
            
            model = clone(self.model_base).set_params(**params_actual)
            model = model.fit(X, y)
            ts_actual = self.perturbation(X, model, p_components, min_max_scaler, count)

            self.model_list.append(model)
            self.min_max_scaler.append(min_max_scaler)
            count = count + 1 
        
        self.train_p_components = p_components

        return self

    def perturbation(self, X, model, p_components, min_max_scaler, count):
        p_i_norm = model.predict(X)
        #p_i = min_max_scaler.inverse_transform(p_i_norm.reshape(-1, 1)).flatten()
        p_i = p_i_norm
        p_i = [None] * self.lag_size * (count+1)  + p_i.tolist()
            
        p_components[f'pert{count}'] = p_i
        
        ts_actual = self.calculate_error_and_new_ts(p_components)
 
        return ts_actual

    def predict_steps(self, ts):
        """ pressupoe contexto de ts e 1 passo a frente"""
        trained_models = zip(
            self.model_list, 
            self.min_max_scaler, 
        )
        ts_actual = ts.copy()
        count = 0
        
        p_components= pd.DataFrame({'actual': ts})
        for model, min_max_scaler in trained_models:
            #ts_norm = min_max_scaler.transform(ts_actual.reshape(-1, 1)).flatten()
            ts_norm = ts_actual
            X, _ = create_window(ts_norm, self.lag_size)

            ts_actual = self.perturbation(X, model, p_components, min_max_scaler, count)

            count = count + 1 
        
        self.predict_steps_p_components = p_components

        return p_components.dropna().drop(columns=['actual']).sum(axis=1).values
        



