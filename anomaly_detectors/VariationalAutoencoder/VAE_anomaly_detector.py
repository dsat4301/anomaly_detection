import math
from collections import OrderedDict
from typing import Callable, Sequence

import mlflow
import numpy as np
import torch
from sklearn.metrics import make_scorer, roc_auc_score
from torch import nn
from torch import optim
from torch.distributions import MultivariateNormal
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss, MSELoss
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from base.base_generative_anomaly_detector import BaseGenerativeNNAnomalyDetector
from base.base_networks import MultivariateGaussianEncoder, Decoder


class VAEAnomalyDetector(BaseGenerativeNNAnomalyDetector):
    def __init__(
            self,
            batch_size: int = 128,
            n_jobs_dataloader: int = 4,
            n_epochs: int = 10,
            device: str = 'cpu',
            scorer: Callable = make_scorer(roc_auc_score, needs_threshold=True),
            learning_rate: float = 1e-4,
            linear: bool = True,
            n_hidden_features: Sequence[int] = None,
            random_state: int = None,
            latent_dimensions: int = 2,
            n_drawings_distributions: int = 1,
            softmax_for_final_decoder_layer: bool = False,
            reconstruction_loss_function: _Loss = MSELoss(reduction='none')):
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

        self.n_drawings_distributions = n_drawings_distributions

    @property
    def offset_(self):
        return self._offset_

    @property
    def _networks(self) -> Sequence[torch.nn.Module]:
        return [self.encoder_network_, self.decoder_network_]

    @property
    def _reset_loss_func(self) -> Callable:
        def reset_loss():
            pass

        return reset_loss

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        X, _ = self._check_ready_for_prediction(X)

        # noinspection PyTypeChecker
        loader = self._get_test_loader(X)

        scores = []
        self.encoder_network_.eval()
        self.decoder_network_.eval()

        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device=self.device)
                mean_encoder, log_variance_encoder = self.encoder_network_(inputs)

                z = self._sample(
                    mean_encoder.repeat_interleave(self.n_drawings_distributions, dim=0),
                    log_variance_encoder.repeat_interleave(self.n_drawings_distributions, dim=0))

                reconstructed_samples = self.decoder_network_(z)
                expected_reconstruction_loss = self.reconstruction_loss_function(
                    inputs.repeat_interleave(self.n_drawings_distributions, dim=0),
                    reconstructed_samples).mean(axis=1)

                expected_reconstruction_loss = torch.reshape(
                    expected_reconstruction_loss,
                    (inputs.size()[0], self.n_drawings_distributions))

                scores += expected_reconstruction_loss.mean(dim=1).cpu().data.numpy().tolist()

        return np.array(scores)

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

        self.loss_ = 0
        self.optimizer_ = optim.Adam(
            list(self.encoder_network_.parameters()) + list(self.decoder_network_.parameters()),
            lr=self.learning_rate)
        self._offset_ = 0

    def _optimize_params(self, inputs: torch.Tensor):
        self.optimizer_.zero_grad()

        mean_encoder, log_variance_encoder = self.encoder_network_(inputs)
        kl_divergence = self._get_kl_divergence(mean_encoder, log_variance_encoder)

        z = self._reparametrize(mean_encoder, log_variance_encoder)

        reconstructed = self.decoder_network_(z)
        inputs_expected = inputs.repeat_interleave(self.n_drawings_distributions, dim=0)
        expected_reconstruction_error = self.reconstruction_loss_function(inputs_expected, reconstructed).mean()

        self.loss_ = kl_divergence + expected_reconstruction_error

        self.loss_.backward()
        self.optimizer_.step()

    @staticmethod
    def _get_kl_divergence(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            torch.mul(
                -0.5,
                torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=1)))

    def _reparametrize(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(torch.mul(0.5, log_variance))
        epsilon = torch.randn(
            size=(mean.size()[0], self.n_drawings_distributions, self.latent_dimensions),
            device=self.device)

        return mean.repeat_interleave(self.n_drawings_distributions, dim=0) + \
               torch.einsum('ij,ikj->ikj', standard_deviation, epsilon).flatten(end_dim=1)

    def _sample(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        covariance_matrix = torch.einsum('ij,jk->ijk', log_variance.exp(), torch.eye(log_variance.size()[1]))
        distribution = MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix)

        return distribution.sample().to(self.device)

    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        mlflow.log_metrics(
            step=epoch,
            metrics=OrderedDict([
                ('Training time', epoch_train_time),
                ('Loss', self.loss_.item())]))
        print(f'Epoch {epoch}/{self.n_epochs},'
              f' Epoch training time: {epoch_train_time},'
              f' Loss: {self.loss_}')
