import math
from collections import OrderedDict
from typing import Callable, Sequence

import mlflow
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import MultivariateNormal
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from base.base_generative_anomaly_detector import BaseGenerativeAnomalyDetector
from base.base_networks import MultivariateGaussianEncoder, Decoder


class VAEAnomalyDetector(BaseGenerativeAnomalyDetector):
    LOG_VARIANCE_LOWER_LIMIT = -80
    LOG_VARIANCE_UPPER_LIMIT = 80
    PRECISION = 5

    def __init__(
            self,
            batch_size: int = 128,
            n_jobs_dataloader: int = 4,
            n_epochs: int = 10,
            device: str = 'cpu',
            scorer: Callable = None,
            learning_rate: float = 1e-4,
            linear: bool = True,
            n_hidden_features: Sequence[int] = None,
            random_state: int = None,
            latent_dimensions: int = 2,
            n_draws_latent_distribution: int = 1,
            softmax_for_final_decoder_layer: bool = False,
            reconstruction_loss_function: _Loss = None):
        super().__init__(
            batch_size,
            n_jobs_dataloader,
            n_epochs,
            device,
            scorer,
            learning_rate,
            linear,
            n_hidden_features,
            random_state,
            novelty=True,
            latent_dimensions=latent_dimensions,
            softmax_for_final_decoder_layer=softmax_for_final_decoder_layer,
            reconstruction_loss_function=reconstruction_loss_function)

        self.n_draws_latent_distribution = n_draws_latent_distribution

        if self.reconstruction_loss_function is not None \
                and self.reconstruction_loss_function.reduction != 'none':
            raise ValueError('Invalid reduction for loss.')

    @property
    def offset_(self):
        return self._offset_

    @property
    def _networks(self) -> Sequence[torch.nn.Module]:
        return [self.encoder_network_, self.decoder_network_]

    @property
    def _reset_loss_func(self) -> Callable:
        def reset_loss():
            self._train_divergence_losses_epoch_ = []
            self._train_reconstruction_losses_epoch_ = []
            self._train_losses_epoch_ = []
            self._validation_losses_epoch_ = []

        return reset_loss

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        X, _ = self._check_ready_for_prediction(X)

        # noinspection PyTypeChecker
        loader = self._get_data_loader(X, shuffle=False)

        scores = []
        self.encoder_network_.eval()
        self.decoder_network_.eval()

        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device=self.device)
                mean_encoder, log_variance_encoder = self.encoder_network_(inputs)

                z = self._sample(
                    mean_encoder.repeat_interleave(self.n_draws_latent_distribution, dim=0),
                    log_variance_encoder.repeat_interleave(self.n_draws_latent_distribution, dim=0))

                reconstructed_samples = self.decoder_network_(z)

                expected_reconstruction_loss = self._get_reconstruction_loss(
                    reconstructed_samples,
                    inputs.repeat_interleave(self.n_draws_latent_distribution, dim=0))

                expected_reconstruction_loss = torch.reshape(
                    expected_reconstruction_loss,
                    (inputs.size()[0], self.n_draws_latent_distribution))

                scores += expected_reconstruction_loss.mean(dim=1).cpu().data.numpy().tolist()

        return np.array(scores).round(self.PRECISION)

    def _get_reconstruction_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        reconstruction_loss_function = self.reconstruction_loss_function \
            if self.reconstruction_loss_function is not None \
            else nn.MSELoss(reduction='none')

        return reconstruction_loss_function(inputs, targets).mean(axis=1)

    def _initialize_fitting(self, train_loader: DataLoader):
        n_hidden_features_fallback = \
            [self.n_features_in_ - math.floor((self.n_features_in_ - self.latent_dimensions) / 2)]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.encoder_network_ = MultivariateGaussianEncoder(
            self.latent_dimensions,
            self.n_features_in_,
            self.n_hidden_features if self.n_hidden_features is not None else n_hidden_features_fallback,
            self.linear)

        self.decoder_network_ = nn.Sequential(nn.Linear(self.latent_dimensions, self.n_features_in_)) \
            if self.linear \
            else Decoder(self.latent_dimensions, self.n_features_in_, self.n_hidden_features, bias=False)

        if self.softmax_for_final_decoder_layer:
            self.decoder_network_.add_module('softmax', nn.Softmax(dim=1))

        self._train_losses_epoch_ = []
        self._train_divergence_losses_epoch_ = []
        self._train_reconstruction_losses_epoch_ = []
        self._validation_losses_epoch_ = []

        self._optimizer_ = optim.Adam(
            list(self.encoder_network_.parameters()) + list(self.decoder_network_.parameters()),
            lr=self.learning_rate)
        self._offset_ = 0

    def _optimize_params(self, inputs: torch.Tensor):
        kl_divergence, expected_reconstruction_errors = self._get_losses(inputs)
        losses = kl_divergence + expected_reconstruction_errors

        self._optimizer_.zero_grad()
        losses.mean().backward()
        self._optimizer_.step()

        self._train_losses_epoch_ += losses.data.numpy().tolist()
        self._train_reconstruction_losses_epoch_ += expected_reconstruction_errors.data.numpy().tolist()
        self._train_divergence_losses_epoch_ += [kl_divergence.item()]

    def _get_losses(self, inputs: torch.Tensor):
        mean_encoder, log_variance_encoder = self.encoder_network_(inputs)

        # cap values to prevent infinity values caused by exponentiation
        log_variance_encoder = self.get_log_variance_with_capped_values(log_variance_encoder)

        # Acc. to Géron, A. (Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow:Concepts, tools
        # and techniques to build intelligent systems (O’Reilly Media, 2019); p. 589),
        # the divergence loss should be scaled, to ensure it has appropriate scale compared to the reconstruction loss.

        kl_divergence = self._get_kl_divergence(mean_encoder, log_variance_encoder) / self.n_features_in_
        z = self._reparametrize(mean_encoder, log_variance_encoder)
        reconstructed = self.decoder_network_(z)

        inputs_expected = inputs.repeat_interleave(self.n_draws_latent_distribution, dim=0)
        expected_reconstruction_errors = self._get_reconstruction_loss(inputs_expected, reconstructed)

        return kl_divergence, expected_reconstruction_errors

    @staticmethod
    def get_log_variance_with_capped_values(log_variance_encoder: torch.Tensor):
        log_variance_encoder[log_variance_encoder > VAEAnomalyDetector.LOG_VARIANCE_UPPER_LIMIT] = \
            VAEAnomalyDetector.LOG_VARIANCE_UPPER_LIMIT
        log_variance_encoder[log_variance_encoder < VAEAnomalyDetector.LOG_VARIANCE_LOWER_LIMIT] = \
            VAEAnomalyDetector.LOG_VARIANCE_LOWER_LIMIT

        return log_variance_encoder

    @staticmethod
    def _get_kl_divergence(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            torch.mul(
                -0.5,
                torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=1)))

    def _reparametrize(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(torch.mul(0.5, log_variance))
        epsilon = torch.randn(
            size=(mean.size()[0], self.n_draws_latent_distribution, self.latent_dimensions),
            device=self.device)

        return mean.repeat_interleave(self.n_draws_latent_distribution, dim=0) + \
               torch.einsum('ij,ikj->ikj', standard_deviation, epsilon).flatten(end_dim=1)

    def _sample(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        log_variance = self.get_log_variance_with_capped_values(log_variance)

        if self.random_state is not None:
            samples = mean + log_variance.exp()
        else:
            covariance_matrix = torch.einsum('ij,jk->ijk', log_variance.exp(), torch.eye(log_variance.size()[1]))
            distribution = MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix)
            samples = distribution.sample().to(self.device)

        return samples

    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        mean_divergence_loss = np.array(self._train_divergence_losses_epoch_).mean()
        mean_reconstruction_loss = np.array(self._train_reconstruction_losses_epoch_).mean()
        mean_training_loss = np.array(self._train_losses_epoch_).mean()

        metrics = OrderedDict([
            ('Training time', epoch_train_time),
            ('Training loss', mean_training_loss),
            ('Training divergence loss', mean_divergence_loss),
            ('Training reconstruction loss', mean_reconstruction_loss)])

        if self._validation_losses_epoch_:
            metrics['Validation loss'] = np.array(self._validation_losses_epoch_).mean()

        mlflow.log_metrics(step=epoch, metrics=metrics)

        print(f'Epoch {epoch}/{self.n_epochs},'
              f' Epoch training time: {epoch_train_time},'
              f' Loss: {self._train_losses_epoch_}')

    def _update_validation_loss_epoch(self, epoch: int, inputs: torch.Tensor):
        kl_divergence, expected_reconstruction_errors = self._get_losses(inputs)

        losses = kl_divergence + expected_reconstruction_errors

        self._validation_losses_epoch_ += losses.data.numpy().tolist()
