from typing import Callable

import numpy as np
import scipy

from base.base_distance_anomaly_detector import BaseDistanceAnomalyDetector


class MahalanobisDistanceAnomalyDetector(BaseDistanceAnomalyDetector):
    """ Anomaly detection based on the Mahalanobis distance.

    Parameters
    ----------
    scorer : Callable, default=None
        Scorer instance to be used in score function.
    random_state : int, default=None
        Seed value to be applied in order to create deterministic results.

    Attributes
    ----------
    covariance_matrix_ : np.ndarray of shape (n_features, n_features)
        The covariance matrix determined by the call of fit.
    inverse_covariance_matrix_ : np.ndarray of shape (n_features, n_features)
        The inverse covariance matrix.
    distribution_center_ : np.ndarray of shape (n_features,)
        The mean values of the training data.

    Examples
    --------
    >>> import numpy
    >>> from anomaly_detectors.distance_based.mahalanobis_distance_anomaly_detector import
    >>>     MahalanobisDistanceAnomalyDetector
    >>> data = numpy.array([[0], [0.44], [0.45], [0.46], [1]])
    >>> md_anomaly_detector = MahalanobisDistanceAnomalyDetector().fit(data)
    >>> md_anomaly_detector.score_samples(data)
    array([1.75596184e+00, 7.15421304e-03, 3.17965024e-03, 7.94912560e-04, 2.23290938e+00])
    """

    def __init__(self, scorer: Callable = None, random_state: int = None):
        super(MahalanobisDistanceAnomalyDetector, self).__init__(scorer, random_state)

    def _initialize_fitting(self, normal_data: np.ndarray):
        if len(normal_data) < 2:
            raise ValueError('Covariance matrix calculation with one sample not possible.')

        if (normal_data.var(axis=0) == 0).any():
            raise ValueError('Feature with zero variance detected. This causes a singular covariance matrix.')

        self.covariance_matrix_ = np.cov(normal_data, rowvar=False)

        if self.n_features_in_ > 1:
            self.inverse_covariance_matrix_ = scipy.linalg.inv(self.covariance_matrix_, check_finite=True)
        else:
            self.inverse_covariance_matrix_ = 1 / self.covariance_matrix_

        self.distribution_center_ = normal_data.mean(axis=0)

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        centered_values = X - self.distribution_center_

        return (np.dot(centered_values, self.inverse_covariance_matrix_) * centered_values).sum(axis=1)

    def _get_tags(self):
        tags = super()._get_tags()
        # noinspection SpellCheckingInspection
        tags['_xfail_checks']['check_estimators_dtypes'] = 'Replaced with customized test.'

        return tags
