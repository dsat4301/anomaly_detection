from parameterized import parameterized

from anomaly_detectors.GANomaly.GANomaly_anomaly_detector import GANomalyAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests
from testing_base.testing_helpers import get_estimator_checks


class GANomalyAnomalyDetectorTests(BaseAnomalyDetectorTests):
    checks = get_estimator_checks(GANomalyAnomalyDetector())

    @property
    def sut(self):
        return GANomalyAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
