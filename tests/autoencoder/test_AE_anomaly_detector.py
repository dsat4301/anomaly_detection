from unittest import TestCase

from parameterized import parameterized

from anomaly_detectors.autoencoder.AE_anomaly_detector import AEAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests


class AEAnomalyDetectorTests(BaseAnomalyDetectorTests, TestCase):
    checks = BaseAnomalyDetectorTests.get_estimator_checks(AEAnomalyDetector())

    def create_sut(self):
        return AEAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
