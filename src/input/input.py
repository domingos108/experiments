import pandas as pd
import numpy as np
from sklearn import preprocessing
from sktime.param_est.stationarity import StationarityKPSS
from statsmodels.tsa.stattools import pacf


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

def open_format_train_val_test(base_name, normalize, lag_size, exec_config, diff_kpss):
    if  (isinstance(base_name, pd.Series)):

        df =  pd.DataFrame({'y': base_name.values})
    else:
        df = load_raw_data(base_name)

    ts_univariate = df['y'].values

    test_size = int(exec_config['test_size']* ts_univariate.shape[0])

    val_size = int(exec_config['val_size'] * ts_univariate.shape[0])
    
    if diff_kpss is True:
        sty_est = StationarityKPSS()  
        sty_est.fit(ts_univariate[0:-test_size]) 

        is_stationary = bool( sty_est.get_fitted_params()["stationary"])
        if is_stationary is False:
            ts_univariate = difference(ts_univariate, m = 1)
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
    
    lag_actual = lag_size

    if lag_size == None:
        lag_actual = get_max_lag_to_consider(ts_univariate, test_size)
        #print(f"PACF LAG SELECT {lag_actual}")

    df_windowed = create_windowing(ts_normalized, lag_actual)
    
    df_train = df_windowed[0:-(test_size+val_size)]
    df_val = df_windowed[-(test_size+val_size): -test_size]
    df_test = df_windowed[-test_size:]

    return ts_univariate, df_train, df_val, df_test, min_max_scaler, test_size, val_size, is_stationary, df['y'].values