import unittest

from parameterized import parameterized
from sklearn.utils.estimator_checks import check_estimator

from anomaly_detectors.SVDD.SVDD_anomaly_detector import SVDDAnomalyDetector
from testing_base.customized_estimator_checks import check_outliers_train, check_outliers_fit_predict


class SVDDAnomalyDetectorTests(unittest.TestCase):
    EXCLUDED_SCIKIT_LEARN_ESTIMATOR_TESTS = ['check_outliers_train', 'check_outliers_fit_predict']

    checks = []
    estimator_checks = check_estimator(SVDDAnomalyDetector(), generate_only=True)

    for estimator, check in estimator_checks:
        name = str(check.func.__name__)
        if name not in EXCLUDED_SCIKIT_LEARN_ESTIMATOR_TESTS:
            checks.append((name, estimator, check))

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)

    @staticmethod
    def test_check_outliers_train():
        sut = SVDDAnomalyDetector()
        check_outliers_train(sut)

    @staticmethod
    def test_check_outliers_fit_predict():
        sut = SVDDAnomalyDetector()
        check_outliers_fit_predict(sut)
