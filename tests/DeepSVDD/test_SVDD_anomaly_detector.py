from parameterized import parameterized

from anomaly_detectors.DeepSVDD.SVDD_anomaly_detector import DeepSVDDAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests
from testing_base.testing_helpers import get_estimator_checks


class DeepSVDDAnomalyDetectorTests(BaseAnomalyDetectorTests):

    checks = get_estimator_checks(DeepSVDDAnomalyDetector())

    def create_sut(self):
        return DeepSVDDAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
