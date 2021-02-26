from abc import abstractmethod
from typing import Callable

from sklearn.metrics import make_scorer, roc_auc_score

from base.base_anomaly_detector import BaseAnomalyDetector

import numpy as np


class BaseDistanceAnomalyDetector(BaseAnomalyDetector):

    def __init__(self, scorer: Callable, random_state: int):
        super(BaseDistanceAnomalyDetector, self).__init__(scorer, random_state)

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

        self._initialize_fitting(normal_data)

        self.__offset_ = 0

        return self

    @abstractmethod
    def _initialize_fitting(self, normal_data: np.ndarray):
        raise NotImplemented
