from unittest import TestCase

from parameterized import parameterized

from anomaly_detectors.variational_autoencoder.VAE_anomaly_detector import VAEAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests


class VAEAnomalyDetectorTests(BaseAnomalyDetectorTests, TestCase):
    checks = BaseAnomalyDetectorTests.get_estimator_checks(VAEAnomalyDetector())

    def create_sut(self):
        return VAEAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
