import numpy as np
from sklearn.base import clone

import config
from model import generics

class GridSearch:
    def __init__(self, 
                 model_class_exp,
                 model,
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
            model_name
        )

    def search_params(self):
        experiment_params = self.experiment_params.copy()
        experiment_params['test_size'] = config.TEST_SIZE
        experiment_params['val_size'] = config.VAL_SIZE
        target_list_mean_metrics = []
        exec_list_metrics = []
        for _ in range(0, self.model_exec): 
            model_exp = self.model_class_exp( 
                clone(self.model),
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
        exec_list_metrics.append(np.mean(exec_list_metrics))
        experiment_params['val_size'] = 0
        predict_results = []
        for _ in range(0, self.model_exec): 
            model_exp_test = self.model_class_exp( 
                clone(self.model),
                self.experiment_id, 
                self.base_name, 
                self.model_name, 
                self.force,
                self.normalize,
                experiment_params
            )
            model_exp_test.fit_predict()
            predict_results.append(model_exp_test)

        generics.save_result(self.fold, self.title, predict_results)

        import ipdb;ipdb.set_trace()
