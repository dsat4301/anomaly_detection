from typing import Callable

import numpy as np
import scipy
from sklearn.metrics import make_scorer, roc_auc_score

from base.base_anomaly_detector import BaseAnomalyDetector


class MahalanobisDistanceAnomalyDetector(BaseAnomalyDetector):

    def __init__(self, scorer: Callable = make_scorer(roc_auc_score, needs_threshold=True), random_state: int = None):
        super(MahalanobisDistanceAnomalyDetector, self).__init__(scorer, random_state)

    @property
    def offset_(self):
        return self.__offset_

    @offset_.setter
    def offset_(self, value: float):
        # noinspection PyAttributeOutsideInit
        self.__offset_ = value

    # noinspection PyPep8Naming, PyAttributeOutsideInit
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        normal_data = self._get_normal_data(X, y)
        self._set_n_features_in(normal_data)

        np.random.RandomState(self.random_state)

        if len(normal_data) < 2:
            raise ValueError('At least two samples are necessary for covariance matrix calculation.')

        self.covariance_matrix_ = np.cov(normal_data, rowvar=False)

        if self.n_features_in_ > 1:
            self.inverse_covariance_matrix_ = scipy.linalg.inv(self.covariance_matrix_, check_finite=True)
        else:
            self.inverse_covariance_matrix_ = 1 / self.covariance_matrix_
        self.distribution_center_ = normal_data.mean(axis=0)
        self.__offset_ = 0

        return self

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        centered_values = X - self.distribution_center_

        return (np.dot(centered_values, self.inverse_covariance_matrix_) * centered_values).sum(axis=1)
