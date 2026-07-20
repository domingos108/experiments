import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator

import config
from model import generics

from sklearn.exceptions import ConvergenceWarning

# Filter out this specific warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def is_not_sklearn(model):
    # returns True if it is NOT a scikit-learn class/instance
    return not isinstance(model, BaseEstimator)


def resolve_lag_size(base_info):
    """
    Resolve o lag_size a usar para uma serie a partir de uma entrada de
    config.BASE_INFORMATION.

    Chave opcional e aditiva: se `fs_lag_size` estiver presente, ela tem
    prioridade (usada pelos experimentos de Feature Selection para expor
    janelas mais profundas de lags aos seletores). Caso contrario, o
    comportamento e identico ao de sempre -- usa `lag_size`, sem alterar
    nenhum dos 5 baselines existentes (ver PLANO_ARQUITETURA.md, Secao 1.3).
    """
    if "fs_lag_size" in base_info:
        return base_info["fs_lag_size"]
    return base_info["lag_size"]


class GridSearch:
    def __init__(self,
                 model_class_exp,
                 model,
                 model_parameters,
                 experiment_id,
                 base_name,
                 model_name,
                 force=True,
                 normalize = True,
                 experiment_params = {},
                 model_exec = 10,
                 use_val_slipt_for_prev = False,
                 save_grid_history = True

        ):
        self.model_class_exp = model_class_exp
        self.model = model
        self.model_parameters = model_parameters
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params
        self.model_exec = model_exec
        self.force = force
        self.normalize = normalize
        self.group_metrics_name = 'val_metrics'
        self.metric = 'RMSE'
        self.use_val_slipt_for_prev = use_val_slipt_for_prev
        self.save_grid_history = save_grid_history
        self.fold, self.title = generics.format_names(
            experiment_id,
            base_name,
            f'{experiment_params["horizon"]}{model_name}'
        )

    def _search_params(self):

        experiment_params = self.experiment_params.copy()
        experiment_params['test_size'] = config.TEST_SIZE
        experiment_params['val_size'] = config.VAL_SIZE
        experiment_params['lag_size'] = resolve_lag_size(config.BASE_INFORMATION[self.base_name])

        target_list_mean_metrics = []
        grid_search_history = []

        model_exec = 1 if self.model_exec < 1 else self.model_exec

        list_params = list(ParameterGrid(self.model_parameters))

        for params in list_params:
            exec_list_metrics = []
            exec_list_all_metrics = []
            # Snapshot ANTES do loop interno: para model_class_exp nao-sklearn,
            # 'params' e o MESMO dict que sera colocado em
            # experiment_params['model_actual_config'] e MUTADO in-place pelo
            # wrapper (ex. neural_forecast_exp.py injeta 'random_seed' e chaves
            # de plumbing) -- capturar depois do loop pegaria a poluicao em vez
            # dos hiperparametros do grid (achado de code-review, Tarefa 3.4).
            params_snapshot = dict(params)

            for _ in range(0, model_exec):
                if is_not_sklearn(self.model):
                    experiment_params['model_actual_config'] = params
                    model_actual = self.model
                else:
                    model_actual = clone(self.model).set_params(** params)

                model_exp = self.model_class_exp(
                    model_actual,
                    self.experiment_id,
                    self.base_name,
                    self.model_name,
                    self.force,
                    self.normalize,
                    experiment_params
                )

                model_exp.fit_predict()
                metrics_results = model_exp.metrics_results
                val_metrics = metrics_results.get(self.group_metrics_name, {self.metric: np.inf})
                exec_list_metrics.append(val_metrics[self.metric])
                if self.save_grid_history:
                    exec_list_all_metrics.append(dict(val_metrics))

            mean_metric = np.mean(exec_list_metrics)
            target_list_mean_metrics.append(mean_metric)

            if self.save_grid_history:
                with np.errstate(invalid="ignore"):
                    # invalid='ignore': std() sobre uma repeticao com np.inf
                    # (sentinela de metrica ausente, linha acima) produz NaN
                    # corretamente -- so silencia o RuntimeWarning de console,
                    # o NaN persistido continua sinalizando o problema real.
                    val_metric_std = float(np.std(exec_list_metrics))
                grid_search_history.append({
                    'params': params_snapshot,
                    'val_metric_mean': float(mean_metric),
                    'val_metric_std': val_metric_std,
                    'val_metric_reps': [float(v) for v in exec_list_metrics],
                    'val_metrics_reps': exec_list_all_metrics,
                })

        int_arg_min = np.argmin(target_list_mean_metrics)

        return target_list_mean_metrics[int_arg_min], list_params[int_arg_min], grid_search_history

    def execution(self):

        best_exec_val, best_params, grid_search_history = self._search_params()

        experiment_params = self.experiment_params.copy()
        experiment_params['test_size'] = config.TEST_SIZE
        experiment_params['lag_size'] = resolve_lag_size(config.BASE_INFORMATION[self.base_name])

        if self.use_val_slipt_for_prev:
            experiment_params['val_size'] = config.VAL_SIZE
        else:
            experiment_params['val_size'] = 0

        predict_results = []
        print(best_params)
        for _ in range(0, self.model_exec): 
            
            if is_not_sklearn(self.model):
                experiment_params['model_actual_config'] = best_params
                model_actual = self.model
            else:
                model_actual = clone(self.model).set_params(** best_params)

            model_exp_test = self.model_class_exp( 
                model_actual,
                self.experiment_id, 
                self.base_name, 
                self.model_name, 
                self.force,
                self.normalize,
                experiment_params
            )
            model_exp_test.fit_predict()

            entry = {'experiment': model_exp_test, 'val_metric': best_exec_val}
            if self.save_grid_history:
                # Mesma referencia de lista em TODAS as model_exec entradas
                # (nao uma copia por entrada) -- proposital: pickle deduplica
                # objetos repetidos por identidade dentro do mesmo dump, entao
                # isso e ~6-8x mais barato em disco que copiar por entrada
                # (medido em code-review, Tarefa 3.4), ao custo de todas as
                # entradas compartilharem o MESMO objeto. Tratar como
                # somente-leitura -- mutar grid_search_history numa entrada
                # afeta todas as outras.
                entry['grid_search_history'] = grid_search_history
            predict_results.append(entry)

        generics.save_result(self.fold, self.title, predict_results)




def grid_seach_multiple_bases(fit_predict_class, model, normalize, model_parameters,
                              experiment_params,
                              model_exec, model_name, experiment_id,
                              force = True,
                              use_val_slipt_for_prev= True,
                              save_grid_history = True
                              ):

    base_name_list = config.BASE_NAME_LIST
    horizon = experiment_params['horizon']
    for base_name in base_name_list:
        print(base_name)

        fold, title = generics.format_names(experiment_id, base_name, f'{horizon}{model_name}')
        if generics.file_exists(title) and (not force):
            continue

        exec_gs = GridSearch(
            fit_predict_class,
            model,
            model_parameters,
            experiment_id,
            base_name,
            model_name,
            force,
            normalize,
            experiment_params,
            model_exec = model_exec,
            use_val_slipt_for_prev = use_val_slipt_for_prev,
            save_grid_history = save_grid_history
        )

        exec_gs.execution()


def load_grid_search_history(pkl_path: Path) -> pd.DataFrame:
    """
    Le o `grid_search_history` persistido por `GridSearch` (Tarefa 3.4) num
    `.pkl` e retorna um DataFrame com uma linha por combinacao testada: uma
    coluna por chave de `params` (ex. `selector__k`), mais
    `val_metric_mean`/`val_metric_std`/`val_metric_reps`/`val_metrics_reps`.
    Colocada aqui (nao em src/utils/) porque le um formato que so
    `GridSearch` produz -- ver PLANO_ARQUITETURA.md Secao 1.6.

    Uso
    ---
        from pathlib import Path
        from model.grid_search_exp import load_grid_search_history

        df = load_grid_search_history(Path('data/result/<experiment_id>/<serie>_<model>.pkl'))
        df.plot(x='selector__k', y='val_metric_mean', yerr='val_metric_std', marker='o')

    Levanta `ValueError` se o `.pkl` nao tiver `grid_search_history` (nao
    presente, ou presente mas vazio) -- ex. gerado com
    `save_grid_history=False`, ou de antes da Tarefa 3.4. Esse historico nao
    e recuperavel retroativamente; re-rode o Grid Search com a instrumentacao
    atual.
    """
    entries = generics.open_saved_result(str(pkl_path))
    entries = entries if isinstance(entries, list) else [entries]

    history = None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("grid_search_history"):
            history = entry["grid_search_history"]
            break

    if not history:
        raise ValueError(
            f"'{pkl_path}' nao tem 'grid_search_history' -- gerado com "
            "save_grid_history=False, ou antes da Tarefa 3.4 (PLANO_ARQUITETURA.md "
            "Secao 1.6). Esse historico nao e recuperavel retroativamente; "
            "re-rode o Grid Search com a instrumentacao atual."
        )

    rows = []
    for combo in history:
        row = dict(combo["params"])
        row["val_metric_mean"] = combo["val_metric_mean"]
        row["val_metric_std"] = combo["val_metric_std"]
        row["val_metric_reps"] = combo["val_metric_reps"]
        row["val_metrics_reps"] = combo.get("val_metrics_reps")
        rows.append(row)

    return pd.DataFrame(rows)