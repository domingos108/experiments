import glob
from sklearn.base import clone
import pandas as pd

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
        break

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

def exec_mean_ensemble(experiment_id, modelo_list):
    exec_config = {
                    "test_size": config.TEST_SIZE,
                    "val_size": config.VAL_SIZE
    }
     
    fold, title = generics.format_names(experiment_id, '', '')

    predict_results = import_exec_models(fold, modelo_list, exec_config)
   
    for serie_name , df_serie in predict_results.groupby('serie_name'):
        exec_serie = []
        for exec_id, df_exec in df_serie.groupby('exec'):

            base_info = input.open_format_train_val_test(
                    serie_name+'.txt', False, None, exec_config, False)
            
            prev_actual = df_exec.pivot(index = 'index', columns='model_name', values='prev').dropna().mean(axis=1)
            metrics_results = metrics.format_metrics_results(prev_actual, base_info)
            model_exp_test = CleanedClass()
            model_exp_test.metrics_results = metrics_results
            exec_serie.append({'experiment': model_exp_test, 'val_metric': None})

        fold, title = generics.format_names(experiment_id, serie_name, 'meanens')

        generics.save_result(fold, title, exec_serie)