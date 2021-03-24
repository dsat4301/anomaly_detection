from abc import abstractmethod
from typing import Callable

import numpy as np

from base.base_anomaly_detector import BaseAnomalyDetector


class BaseDistanceAnomalyDetector(BaseAnomalyDetector):
    """ Base class for distance-based anomaly detectors, implementing BaseAnomalyDetector. """

    def __init__(self, scorer: Callable, random_state: int):
        super(BaseDistanceAnomalyDetector, self).__init__(scorer, random_state)

    @property
    def offset_(self):
        """ Gets the threshold, applied for decision_function.
        :rtype : float
        """
        return self.__offset_

    @offset_.setter
    def offset_(self, value: float):
        """ Sets the threshold, applied for decision_function.
        :param value : float
        """
        # noinspection PyAttributeOutsideInit
        self.__offset_ = value

    # noinspection PyPep8Naming, PyAttributeOutsideInit
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """

        :param X : np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :param y : binary np.ndarray of shape (n_samples,), default=None
            If given, y is used to filter normal values from data. This means only the samples of data
            with the smaller label in y are used for training.
        :param kwargs :
            Ignored. Implemented to comply with the API.

        :return : BaseDistanceAnomalyDetector
            The fitted instance.
        """
        normal_data = self._get_normal_data(X, y)
        self._set_n_features_in(normal_data)

        np.random.RandomState(self.random_state)

        self._initialize_fitting(normal_data)

        self.__offset_ = 0

        return self

    @abstractmethod
    def _initialize_fitting(self, normal_data: np.ndarray):
        raise NotImplemented
