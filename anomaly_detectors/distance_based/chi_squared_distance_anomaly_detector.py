from typing import Callable

import numpy as np

from base.base_distance_anomaly_detector import BaseDistanceAnomalyDetector


class ChiSquaredDistanceAnomalyDetector(BaseDistanceAnomalyDetector):
    """ Anomaly detection based on the Chi-squared distance.

    Parameters
    ----------
    scorer : Callable, default=None
        Scorer instance to be used in score function.
    random_state : int, default=None
        Seed value to be applied in order to create deterministic results.
    alpha : float
        Value applied in the denominator of the Chi-squared distance
        in order to avoid exceptions due to division by zero

    Attributes
    ----------
    distribution_center_ : np.ndarray
        The mean values of the training data.

    Examples
    --------
    >>> import numpy
    >>> from anomaly_detectors.distance_based.chi_squared_distance_anomaly_detector import
    >>>     ChiSquaredDistanceAnomalyDetector
    >>> data = numpy.array([[0], [0.44], [0.45], [0.46], [1]])
    >>> csd_anomaly_detector = ChiSquaredDistanceAnomalyDetector().fit(data)
    >>> csd_anomaly_detector.score_samples(data)
    array([0.2355    , 0.00099451, 0.00071739, 0.00055376, 0.09604422])
    """

    def __init__(self,
                 scorer: Callable = None,
                 random_state: int = None,
                 alpha: float = 1e-3):
        super(ChiSquaredDistanceAnomalyDetector, self).__init__(scorer, random_state)

        self.alpha = alpha

    def _initialize_fitting(self, normal_data: np.ndarray):
        self.distribution_center_ = normal_data.mean(axis=0)

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        """ Return the anomaly score.

        :param X : numpy.ndarray of shape (n_samples, n_features)
            Set of samples to be scored, where n_samples is the number of samples and
            n_features is the number of feature
        :return : numpy.ndarray with shape (n_samples,)
            Array with positive scores.
            Higher values indicate that an instance is more likely to be anomalous.
        """
        return 0.5 * \
               np.sum(np.power(X - self.distribution_center_, 2) / (X + self.distribution_center_) + self.alpha, axis=1)
