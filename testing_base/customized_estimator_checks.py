import numpy as np
from numpy.testing import assert_raises, assert_array_equal, assert_allclose


# noinspection PyPep8Naming
def check_outliers_train(estimator):
    n_samples = 300
    n_features = 2
    X = np.random.random(size=(n_samples, n_features))

    n_samples, n_features = X.shape
    # noinspection PyTypeChecker
    estimator.set_params(random_state=0)

    # fit
    estimator.fit(X)
    # with lists
    estimator.fit(X.tolist())

    y_pred = estimator.predict(X)
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == 'i'

    decision = estimator.decision_function(X)
    scores = estimator.score_samples(X)
    for output in [decision, scores]:
        assert output.dtype == np.dtype('float')
        assert output.shape == (n_samples,)

    # raises error on malformed input for predict
    assert_raises(ValueError, estimator.predict, X.T)

    # decision_function agrees with predict
    dec_pred = (decision >= 0).astype(np.int)
    dec_pred[dec_pred == 0] = -1
    assert_array_equal(dec_pred, y_pred)

    # raises error on malformed input for decision_function
    assert_raises(ValueError, estimator.decision_function, X.T)

    # decision_function is a translation of score_samples
    y_dec = scores - estimator.offset_
    assert_allclose(y_dec, decision)

    # raises error on malformed input for score_samples
    assert_raises(ValueError, estimator.score_samples, X.T)

    # contamination parameter (not for OneClassSVM which has the nu parameter)
    if (hasattr(estimator, 'contamination')
            and not hasattr(estimator, 'novelty')):
        assert True is False, 'Further implementation required if test should be used for outlier detectors.'


# refers to scikit-learn's estimator_checks.check_outliers_fit_predict
# line assert_array_equal(np.unique(y_pred), np.array([-1, 1])) is neither satisfiable nor useful
# noinspection PyPep8Naming
def check_outliers_fit_predict(estimator):
    # Check fit_predict for outlier detectors.

    n_samples = 300
    n_features = 2
    X = np.random.random(size=(n_samples, n_features))
    n_samples, n_features = X.shape

    # noinspection PyTypeChecker
    estimator.set_params(random_state=0)

    y_pred = estimator.fit_predict(X)
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == 'i'

    # check fit_predict = fit.predict when the estimator has both a predict and
    # a fit_predict method. recall that it is already assumed here that the
    # estimator has a fit_predict method
    if hasattr(estimator, 'predict'):
        y_pred_2 = estimator.fit(X).predict(X)
        assert_array_equal(y_pred, y_pred_2)

    if hasattr(estimator, "contamination"):
        assert True is False, 'Further implementation required if test should be used for outlier detectors.'
