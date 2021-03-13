from typing import Callable

import numpy as np

from base.base_distance_anomaly_detector import BaseDistanceAnomalyDetector


class ChiSquareDistanceAnomalyDetector(BaseDistanceAnomalyDetector):
    def __init__(self,
                 scorer: Callable = None,
                 random_state: int = None,
                 alpha: float = 1e-3):
        super(ChiSquareDistanceAnomalyDetector, self).__init__(scorer, random_state)

        self.alpha = alpha

    def _initialize_fitting(self, normal_data: np.ndarray):
        self.distribution_center_ = normal_data.mean(axis=0)

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        return 0.5 * \
               np.sum(np.power(X - self.distribution_center_, 2) / (X + self.distribution_center_) + self.alpha, axis=1)
