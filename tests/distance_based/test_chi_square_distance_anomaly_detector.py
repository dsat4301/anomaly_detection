from unittest import TestCase

from parameterized import parameterized

from anomaly_detectors.distance_based.chi_square_distance_anomaly_detector import ChiSquareDistanceAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests


class ChiSquareDistanceAnomalyDetectorTests(BaseAnomalyDetectorTests, TestCase):
    checks = BaseAnomalyDetectorTests.get_estimator_checks(ChiSquareDistanceAnomalyDetector())

    def create_sut(self):
        return ChiSquareDistanceAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
