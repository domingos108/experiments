import warnings

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.base import RegressorMixin, BaseEstimator

class CustomBaggingRegressor(RegressorMixin, BaseEstimator):
    """
    A custom Bagging Regressor that aggregates predictions using the median.
    """
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, 
                 max_features=1.0, bootstrap=True, bootstrap_features=False, 
                 oob_score=False, warm_start=False, n_jobs=None, 
                 random_state=None, agg_type='mean', verbose=0):
        
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.agg_type = agg_type
        # Internal standard BaggingRegressor (will use mean internally)


    def fit(self, X, y):
        
        self.bagging_base = BaggingRegressor(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            oob_score=self.oob_score,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.bagging_base.fit(X, y)
        # Store the fitted estimators
        self.estimators_ = self.bagging_base.estimators_
        self.cols_to_use = self.bagging_base.estimators_features_
        return self

    def predict(self, X):
        # Get predictions from all individual estimators
        predictions = []
        for estimator,  feat in zip(self.estimators_, self.cols_to_use):
            input_actual = X[:, feat]
            predictions.append(estimator.predict(input_actual))
            
        # Aggregate using the median (axis=0 calculates median across all estimators for each sample)
        if self.agg_type == 'median':
            return np.median(predictions, axis=0)
        
        elif self.agg_type == 'mean':
            return np.mean(predictions, axis=0)
        else:
            raise Exception('agg type not implemented')
