import glob

from sklearn.base import clone
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

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
                try:
                    serie_name = pth.split('/')[-1].split('_')[0]
                    exec_pkl = generics.open_saved_result(pth)
                    lag_size = config.BASE_INFORMATION[f'{serie_name}.txt']['lag_size']
                    exec_atual = exec_config.copy()
                    exec_atual['lag_size'] = lag_size
                    base_info = input.open_format_train_val_test(
                        serie_name+'.txt',  exec_atual)
                    
                    for i, em in enumerate(exec_pkl):
                        #prevs = pd.Series(em['experiment'].metrics_results['train_predict']).to_list()+\
                        #        pd.Series(em['experiment'].metrics_results['val_predict']).to_list() +\
                        #        pd.Series(em['experiment'].metrics_results['test_predict']).to_list()
                        prevs = pd.Series(em['experiment'].metrics_results['val_predict']).to_list() +\
                                pd.Series(em['experiment'].metrics_results['test_predict']).to_list()
                        
                        prevs = [None if abs(p)==np.inf else p for p in prevs]
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
                except:
                    import ipdb;ipdb.set_trace()

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

def lags_recent_dinanic_selection(df, real, ds_args):
    k = ds_args['k']
    lag_max = ds_args.get('lag_size', 1) # Padrão é 1 se não definido

    # Calcula o erro quadrático para todos os pontos
    ts_atu = real[-df.shape[0]:]
    erro_df = df.sub(ts_atu, axis='rows').pow(2)

    # Calcula a média móvel dos erros para considerar o passado (t-1 até t-lag_max)
    # O shift(1) garante que para prever 't', olhamos apenas até 't-1'
    erro_movel = erro_df.rolling(window=lag_max).mean().shift(1)

    all_prevs = []
    
    # Iteramos sobre o dataframe original
    for idx in df.index:
        # Se não houver histórico suficiente para a janela lag_max, pulamos ou tratamos
        if pd.isna(erro_movel.loc[idx]).any():
            all_prevs.append(np.nan) # Ou uma lógica de fallback
            continue

        # Seleciona os k modelos com menor erro médio no histórico (lag_max)
        best_models = erro_movel.loc[idx].sort_values().iloc[0:k].index.to_list()
        
        # Faz a média das previsões desses modelos para o ponto atual
        prevs = df.loc[idx, best_models]
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
    k = ds_args['k']

    ts_atu = real[-df.shape[0]:]
    erro_df = df.sub(ts_atu, axis='rows')
    erro_df = erro_df.pow(2)

    indexes_shift = pd.Series(df.index.to_list()).shift(13)
    all_prevs = []
    for d, df_line in zip(indexes_shift, df.to_dict('records')):
        if pd.isna(d):
            continue

        best_models = erro_df.loc[d].sort_values().iloc[0:k]
        best_models = best_models.index.to_list()
        prevs = [df_line[c] for c in best_models]
        all_prevs.append(np.mean(prevs))

    return all_prevs

def dinanic_selection(df, real, test_size, val_size, ds_args):
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

def fixed_weighting(df, real, test_size, val_size):
    """
    Calcula pesos fixos para cada modelo baseando-se no erro quadrático médio 
    dentro do set de validação e aplica ao set de teste.
    """
    # 1. Alinhamento e extração do erro no set de validação
    ts_atu = real[-df.shape[0]:]
    erro_df = df.sub(ts_atu, axis='rows').pow(2)
    
    # Selecionamos apenas a janela de validação para definir os pesos
    # (A janela que antecede o teste)
    val_error = erro_df.iloc[-(test_size+val_size):-test_size]
    
    # 2. Cálculo do Erro Médio por modelo (MSE Global da Validação)
    mse_global = val_error.mean()
    
    # 3. Cálculo dos Pesos (Inverso do Erro)
    # Modelos com menor erro ganham pesos maiores
    epsilon = 1e-10
    weights = 1.0 / (mse_global + epsilon)
    
    # Normalização: soma dos pesos = 1
    weights_normalized = weights / weights.sum()
    
    # 4. Aplicação dos pesos em todo o dataframe (ou apenas no teste)
    # Multiplicamos cada coluna pelo seu respectivo peso e somamos as linhas
    weighted_forecasts = df.mul(weights_normalized, axis='columns').sum(axis=1)
    
    # Se quiser retornar apenas a parte correspondente ao que seria o 'lag_size' em diante
    # ou o set de teste completo:
    return weighted_forecasts.tolist()

def _predict_single(model, feat, x_values, min_max_scaler, is_bagg):
    """Função auxiliar para processar um único estimador em paralelo."""
    if is_bagg:
        # Seleciona apenas as colunas que este estimador específico usou no treino
        cols_actual = [x_values.columns[f] for f in feat]
        input_actual = x_values[cols_actual]
    else:
        input_actual = x_values
    
    # Realiza a predição e aplica o escalonamento inverso
    prev = model.predict(input_actual.values).reshape(1, -1)
    prev = min_max_scaler.inverse_transform(prev).flatten()
    
    return prev

def predict_estimators(ts_formated, base_models, min_max_scaler):
    x_values = ts_formated.drop(columns=['actual'])
    is_bagg = True
    try:
        cols_to_use = base_models.estimators_features_
    except AttributeError:
        is_bagg = False
        cols_to_use = [None] * len(base_models.estimators_)

    # Execução em paralelo
    # n_jobs=-1 usa todos os núcleos disponíveis
    results = Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(_predict_single)(model, feat, x_values, min_max_scaler, is_bagg)
        for model, feat in zip(base_models.estimators_, base_models.estimators_features_)
    )

    # Constrói o DataFrame final a partir da lista de resultados
    prevs_df = pd.DataFrame(
        {f'model_{i+1}': res for i, res in enumerate(results)}
    )

    return prevs_df

def import_bagging(fold, modelo_list, exec_config):

    execs_models = glob.glob(f'{fold}*_{modelo_list[0]}.*')
    predict_results = []
    
    for pth in execs_models:
        
        serie_name = pth.split('/')[-1].split('_')[0]
        exec_pkl = generics.open_saved_result(pth)

        lag_size = config.BASE_INFORMATION[f'{serie_name}.txt']['lag_size']
        exec_atual = exec_config.copy()
        exec_atual['lag_size'] = lag_size

        base_info = input.open_format_train_val_test(
            serie_name+'.txt',  exec_atual
        )
        df_train = base_info.df_train
        df_val = base_info.df_val
        df_test = base_info.df_test
        min_max_scaler = base_info.min_max_scaler
        
        for i, em in enumerate(exec_pkl):
            
            prevs_df = predict_estimators(pd.concat([df_train, df_val, df_test]),
                                       em['experiment'].model, 
                                       min_max_scaler)
           
            melt_df = prevs_df.reset_index().melt(
                id_vars='index',      
                var_name='model_name',      
                value_name='prev' 
            ) # ajust to the same formar of heterogen ensemble
            real = pd.DataFrame({'real': base_info.original_ts[-prevs_df.shape[0]:]})

            melt_df['serie_name'] = serie_name
            melt_df['exec'] = i
            melt_df = melt_df.merge(real.reset_index(), on='index', how='inner')
            
            predict_results.append(melt_df)

    return pd.concat(predict_results)

    
def exec_ensemble(experiment_id, modelo_list, ens_type, ds_args={},  horizon=1):
    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE,
        'horizon': horizon,
        'diff_kpss': False,
        'normalize': True,
        'type_filter': None
    }
     
    fold, title = generics.format_names(experiment_id, '', '')
    if len(modelo_list)>1: #heterogeneus
        predict_results = import_exec_models(fold, modelo_list, exec_config)
        model_name = f'{horizon}comb{ens_type}'
        model_base = 'hm'
    else: #homogeneus
        predict_results = import_bagging(fold, modelo_list, exec_config)
    
        model_name = f'{modelo_list[0]}{ens_type}'
        model_base = modelo_list[0]

    #predict_results ['index', 'prev', 'model_name', 'serie_name', 'real', 'exec']
    
    for serie_name , df_serie in predict_results.groupby('serie_name'):
        exec_serie = []
        for exec_id, df_exec in df_serie.groupby('exec'):
            
            lag_size = config.BASE_INFORMATION[f'{serie_name}.txt']['lag_size']
            exec_atual = exec_config.copy()
            exec_atual['lag_size'] = lag_size

            base_info = input.open_format_train_val_test(serie_name+'.txt', exec_atual)
            
            df_models = df_exec.pivot(index = 'index', columns='model_name', values='prev').dropna()

            if ens_type == 'mean':
                prev_actual = df_models.mean(axis=1)

            elif ens_type == 'median':
                prev_actual = df_models.median(axis=1).values

            elif ens_type == 'ds':
                prev_actual = dinanic_selection(df_models, base_info.original_ts, base_info.test_size, base_info.val_size, ds_args)
                model_name = f'{model_base}rc{ds_args["rc"]}k{ds_args["k"]}'
            elif ens_type == 'mostrecent':
                prev_actual = lags_recent_dinanic_selection(df_models, base_info.original_ts, ds_args)
                model_name = f'{model_base}lag{ds_args["lag_size"]}k{ds_args["k"]}'
            elif ens_type == 'fixedweights':
                prev_actual= fixed_weighting(df_models, base_info.original_ts, base_info.test_size, base_info.val_size)
            else:
                raise NotImplementedError(f'{ens_type} not implemented')
    
            metrics_results = metrics.format_metrics_results(prev_actual, base_info)
            model_exp_test = CleanedClass()
            model_exp_test.metrics_results = metrics_results
            exec_serie.append({'experiment': model_exp_test, 'val_metric': None})

        fold, title = generics.format_names(experiment_id, serie_name, model_name)

        generics.save_result(fold, title, exec_serie)