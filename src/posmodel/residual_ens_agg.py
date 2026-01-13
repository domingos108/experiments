import pandas as pd
import numpy as np

from model import generics, hybrid_system_exp
from input import input
from metrics import metrics
from posmodel import model_comb

# método principal: exec_ensemble, necessário executar o experimento do hybrid_system_exp.Additive com um ensemble Baggingg
def predict_estimators(ts_formated, base_models, min_max_scaler):
    x_values = ts_formated.drop(columns=['actual'])
    prevs_df = pd.DataFrame()
    cont = 1
    is_bagg = True

    try:
        cols_to_use = base_models.estimators_features_
    except AttributeError:
        is_bagg = False
        cols_to_use= [None] * len(base_models.estimators_)
   
    for model, feat in zip(base_models.estimators_, cols_to_use):
        if is_bagg:
            cols_actual = [x_values.columns[f] for f in feat]
            input_actual = x_values[cols_actual]
        else:
            input_actual = x_values
        
        prev = model.predict(input_actual.values).reshape(1, -1)
        prev = min_max_scaler.inverse_transform(prev).flatten()

        prevs_df['model_' + str(cont)] = prev
        cont = cont + 1

    return prevs_df

def get_conssesuos(line):
    result = (line>=0).value_counts(normalize=True).to_dict()

    if result.get(False, 0) == 0.5:
        return 0
    
    if result.get(False, 0) > 0.5:
        to_consider = line[line <0]
        prev = to_consider.quantile(0.75)
    else:
        to_consider = line[line>=0]
        prev = to_consider.quantile(0.25)

    return prev

def dynamic_selection(prevs_df, linear_prev, experiment_params, base_info):

    df_linear_nonlinear = prevs_df.add(linear_prev, axis=0)
    
    if  experiment_params['ds_args']['rc'] <=1:
        experiment_params['ds_args']['rc']  = int(
            experiment_params['ds_args']['rc'] * base_info.val_size
        )
    all_prevs = model_comb.dinanic_selection(
        df_linear_nonlinear, 
        base_info.original_ts,  
        base_info.test_size, 
        base_info.val_size, 
        experiment_params['ds_args']
    )

    return all_prevs

def oracle(prevs_df, linear_prev, base_info):

    df_linear_nonlinear = prevs_df.add(linear_prev, axis=0)

    all_prevs = model_comb.oracle(
        df_linear_nonlinear, 
        base_info.original_ts,  
    )

    return all_prevs            

def seazonal_dynamic_selection(prevs_df, linear_prev, experiment_params, base_info):
    df_linear_nonlinear = prevs_df.add(linear_prev, axis=0)

    all_prevs = model_comb.seazonal_dinanic_selection(
        df_linear_nonlinear, 
        base_info.original_ts,  
        experiment_params['ds_args']
    )

    return all_prevs

def most_recent_dinanic_selection(prevs_df, linear_prev, experiment_params, base_info):
    df_linear_nonlinear = prevs_df.add(linear_prev, axis=0)

    all_prevs = model_comb.most_recent_dinanic_selection(
        df_linear_nonlinear, 
        base_info.original_ts,  
        experiment_params['ds_args']
    )

    return all_prevs



def exec_ensemble(experiment_params, experiment_id, base_name, model_name):
    select_type = experiment_params['select_type']
    _, title_base = generics.format_names(
                experiment_id, 
                base_name, 
                model_name
    )

    (
        error_series,
        ts_forecast,
        base_info,
        exec_config
    ) = hybrid_system_exp.input_linear_info(
        experiment_id, 
        base_name, 
        experiment_params
    )

    lag_size = base_info.lag_size_formated
    normalize = True
    experiment_params['ds_args']['lag_size'] = lag_size
    base_info_error = input.open_format_train_val_test(pd.Series(error_series), normalize, lag_size, exec_config, False)

    df_train = base_info_error.df_train
    df_val = base_info_error.df_val
    df_test = base_info_error.df_test
    min_max_scaler = base_info_error.min_max_scaler

    exec_pkl = generics.open_saved_result(title_base)
    
    predict_results = []
    
    val_metric = []

    for model_exp in exec_pkl:
        rc = 0
        k = 0
        prevs_df = predict_estimators(pd.concat([df_train, df_val, df_test]), model_exp['experiment'].model, min_max_scaler)
        error_return = True
        if select_type == 'median':
            error_prev = prevs_df.median(axis=1).values

        elif select_type == 'cons':
            error_prev = prevs_df.apply(lambda x: get_conssesuos(x), axis=1)

        elif select_type == 'ds':
            error_return=False
            prev_final = dynamic_selection(prevs_df, ts_forecast[(df_test.shape[1] -1):], experiment_params, base_info)
            rc = experiment_params['ds_args']['rc']
            k = experiment_params['ds_args']['k']

        elif select_type == 'dsseazonal':
            error_return=False
            prev_final = seazonal_dynamic_selection(prevs_df, ts_forecast[(df_test.shape[1] -1):], experiment_params, base_info)
            rc = experiment_params['ds_args']['rc']
            k = experiment_params['ds_args']['k']
            
        elif select_type == 'mostrecent':
            error_return=False
            prev_final = most_recent_dinanic_selection(prevs_df, ts_forecast[(df_test.shape[1] -1):], experiment_params, base_info)
            rc = experiment_params['ds_args']['rc']
            k = experiment_params['ds_args']['k']

        elif select_type == 'oracle':
            error_return=False
            prev_final = oracle(prevs_df, ts_forecast[(df_test.shape[1] -1):], base_info)
            rc = experiment_params['ds_args']['rc']
            k = experiment_params['ds_args']['k']

        else:
            raise NotImplementedError(f'select_type {select_type} not found')
        
        if error_return:
            prev_final = ts_forecast[(df_test.shape[1] -1):] + error_prev

        metrics_results = metrics.format_metrics_results(prev_final, base_info)
        experiment = generics.ResultExp(metrics_results)

        val_metric = metrics_results['val_metrics']['RMSE']

        predict_results.append({'experiment': experiment, 'val_metric': val_metric})

    fold, title = generics.format_names(experiment_id, base_name, f'{model_name}c{select_type}rc{rc}k{k}')

    generics.save_result(fold, title, predict_results)