import math

import hypothesis.strategies as st
import torch
from hypothesis import given
from parameterized import parameterized

from anomaly_detectors.variational_autoencoder.VAE_anomaly_detector import VAEAnomalyDetector
from testing_base.base_anomaly_detector_tests import BaseAnomalyDetectorTests
from testing_base.testing_helpers import get_estimator_checks


class VAEAnomalyDetectorTests(BaseAnomalyDetectorTests):
    checks = get_estimator_checks(VAEAnomalyDetector())
    torch_max_float_value = math.log(torch.finfo().max)
    torch_min_float_value = torch.finfo().min

    def create_sut(self):
        return VAEAnomalyDetector()

    @parameterized.expand(checks)
    def test_scikit_learn_estimator(self, name, estimator, check):
        check(estimator)

    @parameterized.expand([
        (torch.zeros(1, 5), torch.ones(1, 5), 1.7957),
        (torch.zeros(5, 5), torch.ones(5, 5), 1.7957),
        (torch.ones(1, 5), torch.ones(1, 5), 4.2957),
        (torch.ones(5, 5), torch.ones(5, 5), 4.2957)
    ])
    def test_get_kl_divergence(self, mean, log_variance, expected_result):
        result = self.create_sut()._get_kl_divergence(torch.Tensor(mean), torch.Tensor(log_variance)).item()

        self.assertAlmostEqual(expected_result, result, self.precision_places)

    @given(
        st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=torch_min_float_value,
                max_value=torch_max_float_value),
            min_size=5,
            max_size=5),
        st.lists(
            st.floats(min_value=1,
                      allow_nan=False,
                      allow_infinity=False,
                      max_value=torch_max_float_value),
            min_size=5,
            max_size=5))
    def test_get_kl_divergence_property(self, mean, log_variance):
        sut = self.create_sut()

        mean = torch.Tensor([mean])
        log_variance = torch.Tensor([log_variance])

        mean_repeated = mean.repeat_interleave(10, dim=0)
        log_variance_repeated = log_variance.repeat_interleave(10, dim=0)

        self.assertAlmostEqual(
            sut._get_kl_divergence(mean, log_variance).item(),
            sut._get_kl_divergence(mean_repeated, log_variance_repeated).item(),
            self.precision_places)

    @given(
        st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=torch_min_float_value,
                max_value=torch_max_float_value),
            min_size=5,
            max_size=5),
        st.lists(
            st.floats(min_value=1,
                      allow_nan=False,
                      allow_infinity=False,
                      max_value=torch_max_float_value),
            min_size=5,
            max_size=5),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10))
    def test_reparametrize(self, mean, log_variance, n_drawings_distribution, batch_size):
        expected_size = (batch_size * n_drawings_distribution, len(mean))

        sut = self.create_sut()
        # noinspection PyTypeChecker
        sut.set_params(n_drawings_distributions=n_drawings_distribution, latent_dimensions=len(mean))

        result = sut._reparametrize(
            torch.Tensor([mean]).repeat_interleave(batch_size, dim=0),
            torch.Tensor([log_variance]).repeat_interleave(batch_size, dim=0))

        self.assertEqual(result.size(), expected_size)

    @given(
        st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=torch_min_float_value,
                max_value=torch_max_float_value),
            min_size=5,
            max_size=5),
        st.lists(
            st.floats(min_value=1,
                      allow_nan=False,
                      allow_infinity=False,
                      max_value=torch_max_float_value),
            min_size=5,
            max_size=5),
        st.integers(min_value=1, max_value=10))
    def test_sample(self, mean, log_variance, batch_size):
        expected_size = (batch_size, len(mean))

        sut = self.create_sut()
        sut.n_features_in_ = len(mean)

        result = sut._sample(
            torch.Tensor([mean]).repeat_interleave(batch_size, dim=0),
            torch.Tensor([log_variance]).repeat_interleave(batch_size, dim=0))

        self.assertEqual(result.size(), expected_size)
