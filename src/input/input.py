import pandas as pd
from sklearn import preprocessing

import config


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


def open_format_train_val_test(base_name, normalize, lag_size, exec_config):

    df = load_raw_data(base_name)
    ts_univariate = df['y'].values

    test_size = int(exec_config['test_size']* ts_univariate.shape[0])

    val_size = int(exec_config['val_size'] * ts_univariate.shape[0])

    # normalize
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(ts_univariate[0:-(test_size)].reshape(-1, 1))

        ts_normalized = min_max_scaler.transform(ts_univariate.reshape(-1, 1))
        ts_normalized = pd.DataFrame({'y': ts_normalized.flatten()})
    else:
        ts_normalized =  df.copy()
        min_max_scaler = None

    df_windowed = create_windowing(ts_normalized, lag_size)
    
    df_train = df_windowed[0:-(test_size+val_size)]
    df_val = df_windowed[-(test_size+val_size): -test_size]
    df_test = df_windowed[-test_size:]

    return ts_univariate, df_train, df_val, df_test, min_max_scaler, test_size, val_size