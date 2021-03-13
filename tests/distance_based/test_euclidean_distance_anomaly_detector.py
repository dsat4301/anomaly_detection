from unittest import TestCase

from parameterized import parameterized

from anomaly_detectors.distance_based.euclidean_distance_anomaly_detector import EuclideanDistanceAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests


class EuclideanDistanceAnomalyDetectorTests(BaseAnomalyDetectorTests, TestCase):
    checks = BaseAnomalyDetectorTests.get_estimator_checks(EuclideanDistanceAnomalyDetector())

    def create_sut(self):
        return EuclideanDistanceAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
