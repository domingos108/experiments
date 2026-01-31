import pandas as pd
import numpy as np
from sklearn import preprocessing
from sktime.param_est.stationarity import StationarityKPSS
from statsmodels.tsa.stattools import pacf
import pywt


import config

def difference(ts, m = 1):
  diff = []

  for t in range(m,ts.shape[0]):
    value = ts[t] - ts[t-m]
    diff.append(value)

  return np.array(diff)

def inverse_diference(last_ob, value):
  return value + last_ob

def get_inverse_diference(last_ob, value):
  return [inverse_diference(last_ob[i], value[i]) for i in range(0, len(value))]


def load_raw_data(base_name):

    path = config.RAW_DATA_PATH + base_name
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.read_csv(path+'.txt')

    return df


def create_windowing(df, lag_size):
    final_df = None
    for i in range(0, (lag_size + 1)):
        serie = df.shift(i)
        if (i == 0):
            serie.columns = ['actual']
        else:
            serie.columns = [str('lag' + str(i))]
        final_df = pd.concat([serie, final_df], axis=1)

    return final_df.dropna()

def get_max_lag_to_consider(ts_univariate, test_size):
    lag_pacf = pacf(
        ts_univariate[0:-test_size], 
        nlags=20#np.ceil(len(ts_univariate) * 0.1)
    )

    limit = 1.96/np.sqrt(len(ts_univariate[0:-test_size]))
    test_list = [ True if p>limit or p<-limit else False for p in lag_pacf]
    max_lag = max([i for i, val in enumerate(test_list) if val])

    return max_lag

def wavlet_db4(ts_univariate):

    wavelet = 'db4'
    level = 2
    coeffs = pywt.wavedec(ts_univariate, wavelet, level=level)

    # Apply thresholding to the detail coefficients for denoising
    # This is a simple soft thresholding example
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(ts_univariate)))
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)] #

    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[1:]

def exec_filter(ts_univariate, filter_params):
    type_filter = filter_params['type_filter']
    lag_actual = filter_params['lag_actual']

    if type_filter is None:
        return ts_univariate
    
    elif type_filter =='ma':
        return pd.Series(ts_univariate).rolling(lag_actual, min_periods=1).mean().values
    elif type_filter =='db4':
        return wavlet_db4(ts_univariate)
    else:
        raise Exception(f'filter {type_filter} not implemented')

def open_format_train_val_test(base_name, exec_config):
    horizon = exec_config['horizon']
    normalize = exec_config['normalize']
    lag_size = exec_config['lag_size']
    diff_kpss = exec_config['diff_kpss']
    type_filter = exec_config['type_filter']

    
    if  (isinstance(base_name, pd.Series)):
        df =  pd.DataFrame({'y': base_name.values})
        ts_freq = None
        m = None
    else:
        df = load_raw_data(base_name)
        ts_freq = config.BASE_INFORMATION[base_name]['freq']
        m = config.BASE_INFORMATION[base_name]['m']

    ts_univariate = df['y'].values
    test_size = int(exec_config['test_size']* ts_univariate.shape[0])

    val_size = int(exec_config['val_size'] * ts_univariate.shape[0])
    
    lag_actual = lag_size

    if lag_size == 'auto':
        lag_actual = get_max_lag_to_consider(ts_univariate, test_size)
    elif not isinstance(lag_size, int):
        raise Exception(f"LAG TYPE {lag_size} NOT IMPLEMENTED")

    filter_params = {
        'type_filter': type_filter,
        'lag_actual': lag_actual
    }

    ts_univariate = exec_filter(ts_univariate, filter_params)
    
    if diff_kpss is True:
        sty_est = StationarityKPSS()  
        sty_est.fit(ts_univariate[0:-test_size]) 

        is_stationary = bool( sty_est.get_fitted_params()["stationary"])
        if is_stationary is False:
            ts_univariate = difference(ts_univariate, m = horizon)
    else:
        is_stationary = False

    # normalize
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(ts_univariate[0:-(test_size)].reshape(-1, 1))

        ts_normalized = min_max_scaler.transform(ts_univariate.reshape(-1, 1))
        ts_normalized = pd.DataFrame({'y': ts_normalized.flatten()})
    else:
        ts_normalized =  df.copy()
        min_max_scaler = None


    df_windowed = create_windowing(ts_normalized, lag_actual+(horizon-1))
    horizon_cols = [f'actual_{i}' for i in range(1,horizon)] + ['actual']
    lags_cols = [f'lag_{i}' for i in reversed(range(1,lag_actual+1))]
    df_windowed.columns = lags_cols + horizon_cols
    df_windowed.drop(columns = [f'actual_{i}' for i in range(1,horizon)], inplace=True)
    
    df_train = df_windowed[0:-(test_size+val_size)]
    df_val = df_windowed[-(test_size+val_size): -test_size]
    df_test = df_windowed[-test_size:]    

    result = OpenDataOutput(
        ts_univariate, 
        df_train, 
        df_val, 
        df_test, 
        min_max_scaler, 
        test_size, 
        val_size, 
        is_stationary, 
        df,
        lag_actual,
        ts_freq,
        m
    )
    
    return result


class OpenDataOutput:
    def __init__(self, 
                 ts_univariate, 
                 df_train, 
                 df_val, 
                 df_test, 
                 min_max_scaler,
                 test_size, 
                 val_size, 
                 is_stationary, 
                 df,
                 lag_size_formated,
                 freq = 'MS',
                 m = 1
    ):
        self.ts_univariate = ts_univariate
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.min_max_scaler = min_max_scaler
        self.test_size = test_size
        self.val_size = val_size
        self.is_stationary = is_stationary
        self.df = df
        self.original_ts = self.df['y'].values
        self.lag_size_formated = lag_size_formated
        self.freq = freq
        self.m = m

    def sequential_return(self):
        return (
            self.ts_univariate, 
            self.df_train, 
            self.df_val, 
            self.df_test, 
            self.min_max_scaler, 
            self.test_size, 
            self.val_size, 
            self.is_stationary, 
            self.df['y'].values,
            self.lag_size_formated
    )
    def dict_return(self):
        return {
            'ts_univariate': self.ts_univariate,
            'df_train': self.df_train,
            'df_val': self.df_val,
            'df_test': self.df_test,
            'min_max_scaler': self.min_max_scaler,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'is_stationary': self.is_stationary,
            'original_ts': self.df['y'].values,
            'lag_size_formated': self.lag_size_formated
        }
        

