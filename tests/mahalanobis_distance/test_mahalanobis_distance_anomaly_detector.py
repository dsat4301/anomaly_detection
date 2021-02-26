from parameterized import parameterized

from anomaly_detectors.distance_based.mahalanobis_distance_anomaly_detector import \
    MahalanobisDistanceAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests
from testing_base.testing_helpers import get_estimator_checks


class MahalanobisDistanceAnomalyDetectorTests(BaseAnomalyDetectorTests):
    checks = get_estimator_checks(MahalanobisDistanceAnomalyDetector())

    def create_sut(self):
        return MahalanobisDistanceAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)
