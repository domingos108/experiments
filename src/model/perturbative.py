from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn import preprocessing
import pandas as pd
import numpy as np

from input import input
from model import generics
from metrics import metrics

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
        self.learning_rate = 'constant'
        self.fixed_error_lags = True
        self.qtd_perturbations = experiment_params['qtd_perturbations']
        self.earlystop = experiment_params.get('earlystop', None)
        self.bagging_pct = experiment_params.get('bagging_pct', None)

    def calculate_error_and_new_ts(self, p_components):
        df_p_final = p_components.dropna().drop(columns=['actual'])

        max_actual_perts = df_p_final.shape[1] - 1
        if self.learning_rate == 'invscaling':
            props = [self.invscaling(max_actual_perts - i) for i in range(0, max_actual_perts)]
            
        else:
            props = [self.learning_rate_init for i in range(0, max_actual_perts)]
        
        props = [1] + props

        df_p_final = df_p_final * props

        y_actual = p_components.dropna()['actual'].values
        p_final= df_p_final.sum(axis=1).values
        
        error = y_actual - p_final

        absolut_error = np.mean(abs(error))

        return error, absolut_error
    
    def invscaling(self, count):
        return self.learning_rate_init / np.power(count, self.power_t)
   
    def perturbation(self, X, model, p_components, min_max_scaler, count):

        p_i_norm = model.predict(X)
        p_i = min_max_scaler.inverse_transform(p_i_norm.reshape(-1, 1)).flatten()

        p_i_size = ( p_components.shape[0] - p_i.shape[0])
        
        if (p_components.shape[1] > 1 ) and (self.fixed_error_lags is True):
            p_i = ([0] * p_i_size) + p_i.tolist()
        else:
            p_i = ([None] * p_i_size) + p_i.tolist()
        
        p_components = pd.concat([
            p_components, 
            pd.DataFrame({f'pert{count}': p_i})
            ], axis=1)
        
        ts_actual, absolut_error = self.calculate_error_and_new_ts(p_components)
        
        return ts_actual, p_components, absolut_error
        
    def bagging(self, X, y):
        num_linhas = X.shape[0]
        size = int(num_linhas * self.bagging_pct)

        indices_aleatorios = np.random.choice(
            num_linhas,     
            size=size, 
            replace=True  
        )

        return X[indices_aleatorios], y[indices_aleatorios]

    def fit(self, training_ts, base_info:input.OpenDataOutput):
        lag_size_actual = base_info.lag_size_formated

        if base_info.original_ts.ndim > 1:
            raise Exception('y must be a univariate numpy array')
        
        p_components= pd.DataFrame({'actual': training_ts})
        count = 0
        ts_actual = training_ts.copy()
        
        self.error_list = []
        earlystop_count_down = 0
        last_best_id = len(self.perturb_models)
        for model in self.perturb_models:
            
            (
                ts_norm, 
                min_max_scaler
            ) = create_min_max_scale(ts_actual.copy())
            
            X, y = create_window(ts_norm, lag_size_actual)
            if self.bagging_pct is not None:
                X_bagging, y_bagging = self.bagging(X, y)

                model = model.fit(X_bagging, y_bagging)

            else:
                model = model.fit(X, y)
            
            ts_actual, p_components, absolut_error = self.perturbation(X, model, p_components, min_max_scaler, count)
            
            self.min_max_scaler.append(min_max_scaler)
            count = count + 1 
            
            if count>2 and self.earlystop is not None:
                if min(self.error_list[1:]) < absolut_error:
                    earlystop_count_down =  earlystop_count_down + 1
                    if earlystop_count_down>=self.earlystop:
                        break
                else:
                    earlystop_count_down =  0
            self.error_list.append(absolut_error)

        if self.earlystop is not None:
            best_pert_arg = np.argmin(self.error_list[1:]) + 2 ## 1 para por de volta o p0 e outro para pegar o melor index+1
            self.error_list = self.error_list[0: best_pert_arg]

        self.train_p_components = p_components

    def predict_steps(self, base_info:input.OpenDataOutput):
        """ pressupoe contexto de ts e 1 passo a frente"""
        if self.earlystop is not None:
            qt_models_seletected = len(self.error_list )
        else:
            qt_models_seletected = len(self.perturb_models)
                
        trained_models = zip(
            self.perturb_models[0:qt_models_seletected], 
            self.min_max_scaler[0:qt_models_seletected]
        )
        ts_actual = base_info.original_ts.copy()
        count = 0
        
        p_components= pd.DataFrame({'actual': base_info.ts_univariate.copy()})
        lag_size = base_info.lag_size_formated

        for model, min_max_scaler in trained_models:            
        
            ts_norm = min_max_scaler.transform(ts_actual.reshape(-1, 1)).flatten()
            X, _ = create_window(ts_norm, lag_size)

            ts_actual, p_components, _ = self.perturbation(X, model, p_components, min_max_scaler, count)

            count = count + 1 

        self.predict_steps_p_components = p_components
        
        return p_components.dropna().drop(columns=['actual']).sum(axis=1).values
    
    def fit_predict(self):
       
        model_pertub = self.experiment_params['model_pertub']
        pn = []
        if self.qtd_perturbations == None:
            for models_before in self.experiment_params['models_before']:
                _, title = generics.format_names(
                    self.experiment_id, 
                    self.base_name, 
                    models_before
                )
                
                pn.append(generics.open_saved_result(title)[0]['experiment'].model.get_params())
        else:
            for _ in range(1, self.qtd_perturbations):
                pn.append(self.model.get_params())    
        
        lag_size_base = self.experiment_params['lag_size']
        diff_kpss = self.experiment_params.get('diff_kpss', False)

        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size']
        }

        base_info = input.open_format_train_val_test(
            self.base_name, 
            False, 
            lag_size_base, 
            exec_config, 
            diff_kpss
        )
        test_size = base_info.test_size
        val_size = base_info.val_size
        training_ts = base_info.ts_univariate[0:-(test_size+val_size)]
        self.perturb_models = []

        for params_perturb in pn:
            self.perturb_models.append(
                 clone(model_pertub).set_params(** params_perturb)
            )

        self.perturb_models.append(self.model)

        self.fit(training_ts, base_info)
        final_forecast = self.predict_steps(base_info)
        train_predict = final_forecast[0:-(test_size+val_size)]
        val_predict = final_forecast[-(test_size+val_size): -test_size]
        test_predict = final_forecast[-test_size:]
        
        test_metrics = metrics.gerenerate_metric_results(base_info.original_ts[-test_size:], test_predict)

        if val_size!=None and val_size>0:
            val_metrics = metrics.gerenerate_metric_results(base_info.original_ts[-(test_size+val_size): -test_size], val_predict)
        else:
            val_metrics = None

        self.metrics_results = {
            'train_predict': train_predict, 
            'val_predict': val_predict, 
            'test_predict':test_predict,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'time_exec': {'testing': None, 'training': None}
        }
        