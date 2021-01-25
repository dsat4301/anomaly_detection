import unittest
from abc import abstractmethod

from parameterized import parameterized

from testing_base.customized_estimator_checks import check_outliers_train, check_outliers_fit_predict


class BaseAnomalyDetectorTests(unittest.TestCase):
    precision_places = 5

    @abstractmethod
    def create_sut(self):
        raise NotImplemented

    @parameterized.expand([
        (check_outliers_train.__name__, check_outliers_train),
        (check_outliers_fit_predict.__name__, check_outliers_fit_predict)])
    def test_customized_estimator_checks(self, name, check):
        check(self.create_sut())
