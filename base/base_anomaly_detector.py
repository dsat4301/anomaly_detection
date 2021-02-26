from abc import abstractmethod
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class BaseAnomalyDetector(BaseEstimator, OutlierMixin):

    def __init__(
            self,
            scorer: Callable,
            random_state: int = None):
        self.scorer = scorer
        self.random_state = random_state

    @property
    @abstractmethod
    def offset_(self):
        raise NotImplemented

    @offset_.setter
    @abstractmethod
    def offset_(self, value: np.ndarray):
        raise NotImplemented

    # noinspection PyPep8Naming
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        raise NotImplemented

    # noinspection PyPep8Naming
    @abstractmethod
    def score_samples(self, X: np.ndarray):
        raise NotImplemented

    # noinspection PyPep8Naming
    def predict(self, X):
        """ Return the classification result based on the results of decision_function.
        :param X: np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :return: np.ndarray with shape (n_samples,)
            Binary array with -1 indicating an anomaly and 1 indicating normal data.
        """
        X, _ = self._check_ready_for_prediction(X)
        decision_function = self.decision_function(X)
        v_mapping = np.vectorize(lambda x: -1 if x else 1)
        return v_mapping((decision_function <= 0))

    # noinspection PyPep8Naming
    def decision_function(self, X: np.ndarray):
        X, _ = self._check_ready_for_prediction(X)
        anomaly_score = self.score_samples(X)
        return anomaly_score - self.offset_

    # noinspection PyPep8Naming
    def score(self, X: np.ndarray, y: np.ndarray):
        """ Return the score based on the scorer passed as class parameter.
        :param X : np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :param y : np.ndarray of with shape (n_samples,), default=None
            The true anomaly labels.
        :return : float
            Scalar score value.
        """
        X, y = self._check_ready_for_prediction(X, y)

        if self.scorer is None:
            return make_scorer(roc_auc_score, needs_threshold=True)(estimator=self, X=X, y_true=y)

        return self.scorer(estimator=self, X=X, y_true=y)

    # noinspection PyPep8Naming
    def _get_normal_data(self, X, y):
        if y is not None:
            X, y = check_X_y(X, y, estimator=self)
            if len(np.unique(y)) > 2:
                raise ValueError

            normal_data = np.array(X[y == min(y)])
            if len(normal_data) == 0:
                raise ValueError
        else:
            # noinspection PyTypeChecker
            normal_data = np.array(check_array(X))
        return normal_data

    def _set_n_features_in(self, normal_data):
        # noinspection PyAttributeOutsideInit
        self.n_features_in_ = normal_data.shape[1]

    # noinspection PyPep8Naming
    def _check_ready_for_prediction(self, X, y=None):
        check_is_fitted(self)
        if y is not None:
            X, y = check_X_y(X, y)
            y = np.array(y)
        else:
            X = check_array(X)
        X = np.array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid number of features in data.')
        return X, y

    def _more_tags(self):
        return {'binary_only': True}

    @staticmethod
    def get_mapped_prediction(y: np.ndarray):
        return np.vectorize(lambda x: 0 if x == -1 else 1)(y)
