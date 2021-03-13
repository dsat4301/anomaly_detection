from typing import Callable

import numpy as np

from base.base_distance_anomaly_detector import BaseDistanceAnomalyDetector


class EuclideanDistanceAnomalyDetector(BaseDistanceAnomalyDetector):

    def __init__(self, scorer: Callable = None, random_state: int = None):
        super(EuclideanDistanceAnomalyDetector, self).__init__(scorer, random_state)
    
    def _initialize_fitting(self, normal_data: np.ndarray):
        self.distribution_center_ = normal_data.mean(axis=0)

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        return np.sqrt(np.power(X - self.distribution_center_, 2).sum(axis=1))
