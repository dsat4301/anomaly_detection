from typing import Callable

import numpy as np

from base.base_distance_anomaly_detector import BaseDistanceAnomalyDetector


class EuclideanDistanceAnomalyDetector(BaseDistanceAnomalyDetector):
    """ Anomaly detection based on the Euclidean distance.

    Parameters
    ----------
    scorer : Callable, default=None
        Scorer instance to be used in score function.
    random_state : int, default=None
        Seed value to be applied in order to create deterministic results.

    Attributes
    ----------
    distribution_center_ : np.ndarray
        The mean values of the training data.

    Examples
    --------
    >>> import numpy
    >>> from anomaly_detectors.distance_based.euclidean_distance_anomaly_detector import
    >>>     EuclideanDistanceAnomalyDetector
    >>> data = numpy.array([[0], [0.44], [0.45], [0.46], [1]])
    >>> ed_anomaly_detector = EuclideanDistanceAnomalyDetector().fit(data)
    >>> ed_anomaly_detector.score_samples(data)
    array([0.47, 0.03, 0.02, 0.01, 0.53])
    """

    def __init__(self, scorer: Callable = None, random_state: int = None):
        super(EuclideanDistanceAnomalyDetector, self).__init__(scorer, random_state)

    def _initialize_fitting(self, normal_data: np.ndarray):
        self.distribution_center_ = normal_data.mean(axis=0)

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        return np.sqrt(np.power(X - self.distribution_center_, 2).sum(axis=1))
