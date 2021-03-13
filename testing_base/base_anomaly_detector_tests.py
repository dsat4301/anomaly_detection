from abc import abstractmethod

from parameterized import parameterized
from sklearn.utils.estimator_checks import check_estimator

from testing_base.customized_estimator_checks import check_outliers_train, check_outliers_fit_predict


class BaseAnomalyDetectorTests(object):
    precision_places = 5
    customized_estimator_checks = [
        (check_outliers_train.__name__, check_outliers_train),
        (check_outliers_fit_predict.__name__, check_outliers_fit_predict)]

    @abstractmethod
    def create_sut(self):
        raise NotImplemented

    # noinspection PyUnusedLocal
    @parameterized.expand(customized_estimator_checks)
    def test_customized_estimator_checks(self, name, check):
        check(self.create_sut())

    @staticmethod
    def get_estimator_checks(estimator_instance):
        estimator_checks = check_estimator(estimator_instance, generate_only=True)
        return [(str(check.func.__name__), estimator_instance, check) for estimator, check in estimator_checks]
