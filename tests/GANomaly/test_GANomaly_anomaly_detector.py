from unittest import TestCase

from parameterized import parameterized

from anomaly_detectors.GANomaly.GANomaly_anomaly_detector import GANomalyAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests


class GANomalyAnomalyDetectorTests(BaseAnomalyDetectorTests, TestCase):
    checks = BaseAnomalyDetectorTests.get_estimator_checks(GANomalyAnomalyDetector())

    def create_sut(self):
        return GANomalyAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
