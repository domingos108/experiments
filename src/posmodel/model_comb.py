import glob

from sklearn.base import clone
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np

from metrics import metrics
from model import generics
from input import input
import config


class CleanedClass:
    
    def __init__(self):
        self.metrics_results = None


def exec_linear_comb(experiment_id, model_name, model_comb, sufix_to_save):
    fold, _ = generics.format_names(experiment_id, '', '')

    execs_models = glob.glob(f'{fold}*{model_name}*')

    predict_results = []
    for pth in execs_models:
        serie_name = pth.split('/')[-1].split('_')[0]
        exec_pkl = generics.open_saved_result(pth)
        print(pth)

        fold, title = generics.format_names(
                experiment_id, 
                serie_name, 
                model_name+sufix_to_save
        )
        
        for ep in exec_pkl:
            prevs = ep['experiment'].df_pert.dropna().drop(columns='actual')

            real = ep['experiment'].df_pert.dropna()['actual']
            test_size = ep['experiment'].base_info.test_size

            x_train = prevs.iloc[0:-test_size]
            y_train = real.iloc[0:-test_size]
            
            clf = clone(model_comb)
            clf.fit(x_train, y_train)
            prevs_comb = clf.predict(prevs)
            metrics_results = metrics.format_metrics_results(prevs_comb, ep['experiment'].base_info)
            model_exp_test = CleanedClass()
            model_exp_test.metrics_results = metrics_results
            predict_results.append({'experiment': model_exp_test, 'val_metric': None})

        generics.save_result(fold, title, predict_results)


def import_exec_models(fold, modelo_list, exec_config):
    predict_results = []

    for model_name  in modelo_list:
            execs_models = glob.glob(f'{fold}*_{model_name}.*')
            
            for pth in execs_models:
                serie_name = pth.split('/')[-1].split('_')[0]
                exec_pkl = generics.open_saved_result(pth)
               
                base_info = input.open_format_train_val_test(
                    serie_name+'.txt', False, None, exec_config, False)
                
                for i, em in enumerate(exec_pkl):
                    prevs = pd.Series(em['experiment'].metrics_results['train_predict']).to_list()+\
                            pd.Series(em['experiment'].metrics_results['val_predict']).to_list() +\
                            pd.Series(em['experiment'].metrics_results['test_predict']).to_list()
                    to_fill =  [None] * (len(base_info.original_ts) - len(prevs))
                    
                    actual_exec = pd.DataFrame({
                        'prev': to_fill + prevs,
                        'model_name': model_name,
                        'serie_name': serie_name,
                    'real':  base_info.original_ts,
                        'exec': i
                    
                    }
                    )
                    predict_results.append(actual_exec.reset_index())

    return pd.concat(predict_results)


def oracle(df, real):
    ts_atu = real[-df.shape[0]:]
    erro_df = df.sub(ts_atu, axis='rows')
    erro_df = erro_df.pow(2)
    indexes = pd.Series(df.index.to_list())

    all_prevs = []
    for d, df_line in zip(indexes, df.to_dict('records')):

        best_models = erro_df.loc[d].sort_values().iloc[[0]]
        best_models = best_models.index.to_list()
        prevs = [df_line[c] for c in best_models]
        all_prevs.append(np.mean(prevs))

    return all_prevs

def most_recent_dinanic_selection(df, real,  ds_args):
    k = ds_args['k']

    ts_atu = real[-df.shape[0]:]
    erro_df = df.sub(ts_atu, axis='rows')
    erro_df = erro_df.pow(2)

    indexes_shift = pd.Series(df.index.to_list()).shift(1)
    all_prevs = []
    for d, df_line in zip(indexes_shift, df.to_dict('records')):
        if pd.isna(d):
            continue

        best_models = erro_df.loc[d].sort_values().iloc[0:k]
        best_models = best_models.index.to_list()
        prevs = [df_line[c] for c in best_models]
        all_prevs.append(np.mean(prevs))

    return all_prevs

def seazonal_dinanic_selection(df, real,  ds_args):
    rc = ds_args['rc']
    lag_size = ds_args['lag_size']
    k = ds_args['k']

    ts_atu = real[-df.shape[0]:]
    erro_df = df.sub(ts_atu, axis='rows')
    erro_df = erro_df.pow(2)

    indexes_shift = pd.Series(df.index.to_list()).shift(12)
    all_prevs = []
    for d, df_line in zip(indexes_shift, df.to_dict('records')):
        if pd.isna(d):
            continue

        best_models = erro_df.loc[d].sort_values().iloc[0:k]
        best_models = best_models.index.to_list()
        prevs = [df_line[c] for c in best_models]
        all_prevs.append(np.mean(prevs))

    return all_prevs

def dinanic_selection(df,real, test_size, val_size, ds_args):
    rc = ds_args['rc']
    lag_size = ds_args['lag_size']
    k = ds_args['k']

    ts_atu = real[-df.shape[0]:]
    erro_df = df.sub(ts_atu, axis='rows')
    erro_df = erro_df.pow(2)
    val_error = erro_df.iloc[-(test_size+val_size):-test_size]
    val_error.reset_index(drop=True, inplace=True)

    ts_windowed = input.create_windowing(lag_size=lag_size,
                                    df=pd.DataFrame({'actual': ts_atu}))

    ts_windowed.drop(columns=['actual'], inplace=True)

    val_windowed = ts_windowed.iloc[-(test_size+val_size):-test_size]
    val_windowed.reset_index(drop=True, inplace=True)

    dists = euclidean_distances(ts_windowed, val_windowed)

    df = df.iloc[lag_size:]
    all_prevs = []
    
    for d, df_line in zip(dists, df.to_dict('records')):
        d_rc = np.argsort(d)[0:rc]
        error_rc = val_error.loc[list(d_rc)]
        best_models = error_rc.mean().sort_values().iloc[0:k]
        best_models = best_models.index.to_list()
        prevs = [df_line[c] for c in best_models]
        all_prevs.append(np.mean(prevs))

    return all_prevs

def exec_ensemble(experiment_id, modelo_list, ens_type, ds_args={},  horizon=1):
    exec_config = {
                    "test_size": config.TEST_SIZE,
                    "val_size": config.VAL_SIZE,
                    'horizon': horizon
    }
     
    fold, title = generics.format_names(experiment_id, '', '')

    predict_results = import_exec_models(fold, modelo_list, exec_config)
    model_name = f'comb{ens_type}'

    for serie_name , df_serie in predict_results.groupby('serie_name'):
        exec_serie = []
        for exec_id, df_exec in df_serie.groupby('exec'):

            base_info = input.open_format_train_val_test(
                    serie_name+'.txt', False, None, exec_config, False)
            
            df_models = df_exec.pivot(index = 'index', columns='model_name', values='prev').dropna()

            if ens_type == 'mean':
                prev_actual = df_models.mean(axis=1)

            elif ens_type == 'median':
                prev_actual = df_models.median(axis=1).values

            elif ens_type == 'ds':
                prev_actual = dinanic_selection(df_models, base_info.original_ts, base_info.test_size, base_info.val_size, ds_args)
                model_name = f'comb{ens_type}rc{ds_args["rc"]}k{ds_args["k"]}'

            else:
                raise NotImplementedError(f'{ens_type} not implemented')
    
            metrics_results = metrics.format_metrics_results(prev_actual, base_info)
            model_exp_test = CleanedClass()
            model_exp_test.metrics_results = metrics_results
            exec_serie.append({'experiment': model_exp_test, 'val_metric': None})

        fold, title = generics.format_names(experiment_id, serie_name, model_name)

        generics.save_result(fold, title, exec_serie)