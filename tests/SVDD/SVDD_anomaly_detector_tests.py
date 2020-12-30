from parameterized import parameterized

from anomaly_detectors.SVDD.SVDD_anomaly_detector import SVDDAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests
from testing_base.testing_helpers import get_estimator_checks


class SVDDAnomalyDetectorTests(BaseAnomalyDetectorTests):

    checks = get_estimator_checks(SVDDAnomalyDetector())

    @property
    def sut(self):
        return SVDDAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
