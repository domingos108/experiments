import copy

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator

import config
from model import generics

def is_not_sklearn(model):
    # returns True if it is NOT a scikit-learn class/instance
    return not isinstance(model, BaseEstimator)
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
                 model_exec = 10

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

        self.fold, self.title = generics.format_names(
            experiment_id, 
            base_name, 
            f'{experiment_params["horizon"]}{model_name}'
        )

    def _search_params(self):

        experiment_params = self.experiment_params.copy()
        experiment_params['test_size'] = config.TEST_SIZE
        experiment_params['val_size'] = config.VAL_SIZE

        target_list_mean_metrics = []

        model_exec = 1 if self.model_exec < 1 else self.model_exec

        list_params = list(ParameterGrid(self.model_parameters))

        for params in list_params:
            exec_list_metrics = []

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
                exec_list_metrics.append(
                    metrics_results.get(self.group_metrics_name, {self.metric: np.inf})[self.metric]
                )

            target_list_mean_metrics.append(np.mean(exec_list_metrics))

        int_arg_min = np.argmin(target_list_mean_metrics)
        
        return target_list_mean_metrics[int_arg_min], list_params[int_arg_min]

    def execution(self):
        
        best_exec_val, best_params = self._search_params()

        experiment_params = self.experiment_params.copy()
        experiment_params['test_size'] = config.TEST_SIZE
        experiment_params['val_size'] = 0
        predict_results = []
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
            
            predict_results.append({'experiment': model_exp_test, 'val_metric': best_exec_val})
        
        generics.save_result(self.fold, self.title, predict_results)




def grid_seach_multiple_bases(fit_predict_class, model, normalize, model_parameters,
                              experiment_params,
                              model_exec, model_name, experiment_id, 
                              force = True):

    base_name_list = config.BASE_NAME_LIST

    for base_name in base_name_list:
        print(base_name)
        fold, title = generics.format_names(experiment_id, base_name, model_name)

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
            model_exec = model_exec
        )

        exec_gs.execution()