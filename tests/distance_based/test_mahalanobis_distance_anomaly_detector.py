from unittest import TestCase

from parameterized import parameterized

from anomaly_detectors.distance_based.mahalanobis_distance_anomaly_detector import \
    MahalanobisDistanceAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests
from testing_base.customized_estimator_checks import check_estimators_dtypes


class MahalanobisDistanceAnomalyDetectorTests(BaseAnomalyDetectorTests, TestCase):
    checks = BaseAnomalyDetectorTests.get_estimator_checks(MahalanobisDistanceAnomalyDetector())
    customized_estimator_checks = BaseAnomalyDetectorTests.customized_estimator_checks + \
                                  [(check_estimators_dtypes.__name__, check_estimators_dtypes)]

    def create_sut(self):
        return MahalanobisDistanceAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)

    # noinspection PyUnusedLocal
    @parameterized.expand(customized_estimator_checks)
    def test_customized_estimator_checks(self, name, check):
        check(self.create_sut())
