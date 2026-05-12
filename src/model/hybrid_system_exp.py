import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.base import clone

from input import input
from model import generics
from metrics import metrics


class ResultExp:
    def __init__(self, metrics_results):
        self.metrics_results = metrics_results

def invscaling(count, power_t, learning_rate_init):
    return learning_rate_init / np.power(count, power_t)

def input_linear_info(experiment_id, base_name, experiment_params):

    linear_fold, linear_title = generics.format_names(
        experiment_id, 
        base_name, 
        experiment_params['linear_model_name']
    )

    exec_config = {
        "test_size": experiment_params['test_size'],
        "val_size": experiment_params['val_size'],
        "horizon": experiment_params['horizon'],
         'lag_size': experiment_params['lag_size'],
         'diff_kpss': experiment_params['diff_kpss'],
         'normalize': False,
         'type_filter': None

    }   
 
    base_info = input.open_format_train_val_test( base_name, exec_config)
    
    pn = generics.open_saved_result(
        linear_title
    )[0]['experiment'].metrics_results

    if pn['val_predict'] is not None:
        ts_forecast = np.concatenate((pn['train_predict'],  pn['val_predict'], pn['test_predict']), axis=0)
    else:
        ts_forecast = np.concatenate((pn['train_predict'], pn['test_predict']), axis=0)
    
    if pn.get('residual_series', None) is None:

        if( ts_forecast.shape[0] != base_info.original_ts.shape[0] ):
            raise Exception("Size of linear model forecast must be the same size of the time series")
    
        error_series = np.subtract(base_info.original_ts, ts_forecast)
    else:
        error_series = pn['residual_series']
    
    return (
        error_series[base_info.lag_size_formated:],
        ts_forecast[base_info.lag_size_formated:],
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
            'time_exec': residual_result['time_exec'],
            'linear_forecast': ts_forecast[lag_size:] ,
            'nonlinear_forecast': all_residual_forecasts
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

class HighLowAdditive:
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




class ResidualCombination:
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
    
    def exec_comb(self):
        lag_size_base = self.experiment_params['lag_size']
        diff_kpss = self.experiment_params['diff_kpss']
        exec_config = {
            "test_size": self.experiment_params['test_size'],
            "val_size": self.experiment_params['val_size'],
            'horizon': self.experiment_params['horizon']
        }

        nonlinear_fold, nonlinear_title = generics.format_names(
            self.experiment_id, 
            self.base_name, 
            self.experiment_params['nonlinear_model']
        )

        fold, title = generics.format_names(
            self.experiment_id, 
            self.base_name, 
            self.model_name
        )

        list_execs = generics.open_saved_result(
            nonlinear_title
        )

        base_info = input.open_format_train_val_test(
            self.base_name, 
            False, 
            lag_size_base, 
            exec_config, 
            diff_kpss
        )
        predict_results = []
        for le in list_execs:
            ts_size = le['experiment'].metrics_results['nonlinear_forecast'].shape[0]
            df_input = pd.DataFrame({
                    'linear_forecast': le['experiment'].metrics_results['linear_forecast'],
                    'nonlinear_forecast': le['experiment'].metrics_results['nonlinear_forecast']
                    })
            
            df_input['ens']= df_input['linear_forecast'] + df_input['nonlinear_forecast']
            df_output = pd.DataFrame({'actual': base_info.ts_univariate[-ts_size:].copy()})

            x_train, y_train, x_val, y_val, x_test, min_max_scaler_y = self.split_training_test(df_input, df_output, base_info)
            (
                train_predict, 
                val_predict, 
                test_predict,
                time_exec
            ) = generics.fit_predict_ml_schemma(self.model, x_train, y_train, x_val, x_test)
            
            test_size = base_info.test_size
            val_size = base_info.val_size
            metric_result = generics.format_forecats(base_info.original_ts, 
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
                        test_predict
            )

            model_exp_test = ResultExp(metric_result)

            predict_results.append({'experiment': model_exp_test, 'val_metric': np.inf})

        generics.save_result(fold, title, predict_results)


class KhasheiBijariHybrid:
    """
    Arquitetura híbrida não-linear proposta por Khashei & Bijari (2011).
    
    Diferente da abordagem aditiva de Zhang (ARIMA + ML_residuos), esta classe
    treina uma única rede neural que recebe entrada combinada:
    
        y_t = f(e_{t-1}, ..., e_{t-n1}, L̂_t, z_{t-1}, ..., z_{t-m1})
    
    onde:
        e_{t-k}  = resíduos do ARIMA em t-k
        L̂_t     = previsão linear (ARIMA) no tempo t
        z_{t-k}  = série original pré-processada em t-k
        y_t      = valor real da série no tempo t (target)
    
    Referência
    ----------
    Khashei, M., & Bijari, M. (2011). "A novel hybridization of artificial
    neural networks and ARIMA models for time series forecasting."
    Applied Soft Computing, 11(2), 2664-2675.
    """
    
    def __init__(self, 
                 model,
                 experiment_id, 
                 base_name, 
                 model_name, 
                 force=True,
                 normalize=True,
                 experiment_params={}
        ):
        """
        Parâmetros
        ----------
        experiment_params : dict
            Deve conter:
                'n1_lags' : int
                    Número de lags dos resíduos ARIMA a usar (e_{t-1}, ..., e_{t-n1})
                'm1_lags' : int
                    Número de lags da série original a usar (z_{t-1}, ..., z_{t-m1})
                'linear_model_name' : str
                    Nome do modelo ARIMA pré-treinado (ex: 'arima')
        """
        self.model = model
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params
        self.force = force
        self.normalize = normalize
    
    def format_khashei_bijari_input(
        self, 
        ts_univariate,      # série original pré-processada
        error_series,       # resíduos ARIMA
        ts_forecast,        # previsão ARIMA
        n1_lags,            # lags dos resíduos
        m1_lags,            # lags da série original
        is_stationary
    ):
        """
        Constrói a super-matriz de entrada combinada e o target conforme
        arquitetura de Khashei & Bijari (2011).
        
        Entrada X: [e_{t-1}, ..., e_{t-n1}, L̂_t, z_{t-1}, ..., z_{t-m1}]
        Target y:  y_t (valor real da série no tempo t)
        
        Retorna
        -------
        df_input : pd.DataFrame
            Matriz de features combinadas (n_samples × (n1 + 1 + m1))
        df_output : pd.DataFrame
            Target (valor real da série no tempo t)
        """
        # --- 1. Lags dos resíduos ARIMA: e_{t-1}, ..., e_{t-n1} ---
        if not is_stationary:
            error_series_adj = [0] + list(error_series)
        else:
            error_series_adj = error_series
        
        error_wind = input.create_windowing(pd.DataFrame(error_series_adj), n1_lags)
        error_cols = [f'residual_lag_{i}' for i in reversed(range(1, n1_lags + 1))] + ['actual']
        error_wind.columns = error_cols
        # Remove coluna 'actual' (que seria e_t), mantém apenas lags
        error_lags = error_wind.drop(columns=['actual'])
        
        # --- 2. Previsão linear no tempo t: L̂_t ---
        # ts_forecast já está alinhado após lag_size_formated no input_linear_info
        linear_pred = pd.DataFrame({'linear_forecast_t': ts_forecast})
        
        # --- 3. Lags da série original: z_{t-1}, ..., z_{t-m1} ---
        if not is_stationary:
            ts_univariate_adj = [0] + list(ts_univariate)
        else:
            ts_univariate_adj = ts_univariate
        
        series_wind = input.create_windowing(pd.DataFrame(ts_univariate_adj), m1_lags)
        series_cols = [f'series_lag_{i}' for i in reversed(range(1, m1_lags + 1))] + ['actual']
        series_wind.columns = series_cols
        # Mantém apenas lags, não o valor atual
        series_lags = series_wind.drop(columns=['actual'])
        
        # --- 4. Target: valor real da série no tempo t (y_t) ---
        df_output = series_wind[['actual']]  # y_t
        
        # --- 5. Concatenação horizontal: [resíduo_lags | L̂_t | série_lags] ---
        # Importante: error_wind e series_wind têm tamanhos diferentes se n1 ≠ m1
        # Precisamos alinhar pelo menor conjunto
        max_lag = max(n1_lags, m1_lags)
        
        # Ajusta todos ao mesmo tamanho (cortando início se necessário)
        if n1_lags < max_lag:
            diff = max_lag - n1_lags
            error_lags = error_lags.iloc[diff:].reset_index(drop=True)
        if m1_lags < max_lag:
            diff = max_lag - m1_lags
            series_lags = series_lags.iloc[diff:].reset_index(drop=True)
        
        # Alinha linear_pred e df_output ao tamanho correto
        # linear_pred e df_output vêm de create_windowing com lag=max_lag equivalente
        # Na prática, ts_forecast e series_wind['actual'] já estão do tamanho correto
        min_len = min(len(error_lags), len(linear_pred), len(series_lags))
        
        error_lags = error_lags.iloc[-min_len:].reset_index(drop=True)
        linear_pred = linear_pred.iloc[-min_len:].reset_index(drop=True)
        series_lags = series_lags.iloc[-min_len:].reset_index(drop=True)
        df_output = df_output.iloc[-min_len:].reset_index(drop=True)
        
        df_input = pd.concat([error_lags, linear_pred, series_lags], axis=1)
        
        return df_input, df_output
    
    def split_training_test(self, df_input, df_output, base_info):
        """
        Split treino/validação/teste e normalização Min-Max (se habilitada).
        
        Normalização é aplicada apenas com base no conjunto de treino para
        evitar data leakage.
        """
        df_input_norm = df_input.copy()
        df_output_norm = df_output.copy()
        min_max_scaler_y = None
        
        if self.normalize:
            min_max_scaler_x = preprocessing.MinMaxScaler()
            min_max_scaler_y = preprocessing.MinMaxScaler()
            
            # Fit apenas no treino
            train_end_idx = -(base_info.test_size + base_info.val_size)
            if train_end_idx == 0:
                train_end_idx = len(df_input)
            
            min_max_scaler_x.fit(df_input.iloc[:train_end_idx])
            min_max_scaler_y.fit(df_output.iloc[:train_end_idx])
            
            df_input_norm = pd.DataFrame(
                min_max_scaler_x.transform(df_input),
                columns=df_input.columns
            )
            df_output_norm = pd.DataFrame(
                min_max_scaler_y.transform(df_output),
                columns=df_output.columns
            )
        
        # Split
        test_size = base_info.test_size
        val_size = base_info.val_size
        
        y_train = df_output_norm.iloc[:-(test_size + val_size)].values.flatten()
        x_train = df_input_norm.iloc[:-(test_size + val_size)].values
        
        y_val = df_output_norm.iloc[-(test_size + val_size):-test_size].values.flatten()
        x_val = df_input_norm.iloc[-(test_size + val_size):-test_size].values
        
        x_test = df_input_norm.iloc[-test_size:].values
        
        return x_train, y_train, x_val, y_val, x_test, min_max_scaler_y
    
    def fit_predict(self):
        """
        Pipeline completo:
        1. Carrega resíduos e previsões do ARIMA pré-treinado
        2. Constrói super-matriz de entrada combinada
        3. Treina modelo ML
        4. Inverte transformações
        5. Calcula métricas
        """
        # --- 1. Carrega dados do ARIMA ---
        (
            error_series,
            ts_forecast,
            base_info,
            exec_config
        ) = input_linear_info(
            self.experiment_id, 
            self.base_name, 
            self.experiment_params, 
        )
        
        original_ts = base_info.original_ts
        test_size = base_info.test_size
        val_size = base_info.val_size
        
        # --- 2. Parâmetros de lags (padrão se não especificado) ---
        n1_lags = self.experiment_params.get('n1_lags', base_info.lag_size_formated)
        m1_lags = self.experiment_params.get('m1_lags', base_info.lag_size_formated)
        
        # --- 3. Construção da super-matriz de entrada ---
        df_input, df_output = self.format_khashei_bijari_input(
            base_info.ts_univariate,
            error_series,
            ts_forecast,
            n1_lags,
            m1_lags,
            base_info.is_stationary
        )
        
        # --- 4. Split e normalização ---
        (
            x_train, y_train, x_val, y_val, x_test, min_max_scaler_y
        ) = self.split_training_test(df_input, df_output, base_info)
        
        # --- 5. Treino e predição ---
        (
            train_predict, 
            val_predict, 
            test_predict,
            time_exec
        ) = generics.fit_predict_ml_schemma(
            self.model, 
            x_train, 
            y_train, 
            x_val, 
            x_test
        )
        
        # --- 6. Inversão de transformações ---
        self.metrics_results = generics.format_forecats(
            original_ts, 
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
            test_predict
        )