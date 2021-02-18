from sklearn.utils.estimator_checks import check_estimator

from anomaly_detectors.MahalanobisDistance.mahalanobis_distance_anomaly_detector import \
    MahalanobisDistanceAnomalyDetector

EXCLUDED_SCIKIT_LEARN_ESTIMATOR_TESTS = ['check_outliers_train', 'check_outliers_fit_predict']
EXCLUDED_SCIKIT_LEARN_ESTIMATOR_TESTS_MAHALANOBIS = ['check_fit2d_1sample']


def get_estimator_checks(estimator_instance):

    checks = []
    estimator_checks = check_estimator(estimator_instance, generate_only=True)

    for estimator, check in estimator_checks:
        name = str(check.func.__name__)
        if name not in EXCLUDED_SCIKIT_LEARN_ESTIMATOR_TESTS\
                and (not isinstance(estimator, MahalanobisDistanceAnomalyDetector)
                     or name not in EXCLUDED_SCIKIT_LEARN_ESTIMATOR_TESTS_MAHALANOBIS):
            checks.append((name, estimator, check))

    return checks
