from collections import OrderedDict
from typing import Callable, Sequence

import mlflow
import numpy as np
import torch
from sklearn.metrics import make_scorer, roc_auc_score
from torch import nn, optim
from torch.nn import Softmax
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from base.base_generative_anomaly_detector import BaseGenerativeNNAnomalyDetector
from base.base_networks import Encoder, Decoder


class AEAnomalyDetector(BaseGenerativeNNAnomalyDetector):
    PRECISION = 5

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
            softmax_for_final_decoder_layer: bool = False,
            reconstruction_loss_function: _Loss = nn.MSELoss(reduction='none')):
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
            reconstruction_loss_function=reconstruction_loss_function)

        self.softmax_for_final_decoder_layer = softmax_for_final_decoder_layer

    @property
    def offset_(self):
        return self._offset_

    @property
    def _networks(self) -> Sequence[torch.nn.Module]:
        return [self.encoder_network_, self.decoder_network_]

    @property
    def _reset_loss_func(self) -> Callable:
        def func():
            pass

        return func

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
                reconstructed = self.decoder_network_(self.encoder_network_(inputs))
                anomaly_scores = self.reconstruction_loss_function(inputs, reconstructed).mean(axis=1)
                scores += anomaly_scores.cpu().data.numpy().tolist()

        return np.array(scores).round(self.PRECISION)

    def _initialize_fitting(self, train_loader: DataLoader):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.encoder_network_ = nn.Sequential(nn.Linear(self.n_features_in_, self.latent_dimensions)) \
            if self.linear \
            else Encoder(self.latent_dimensions, self.n_features_in_, self.n_hidden_features, bias=True)

        self.decoder_network_ = nn.Sequential(nn.Linear(self.latent_dimensions, self.n_features_in_)) \
            if self.linear \
            else Decoder(self.latent_dimensions, self.n_features_in_, self.n_hidden_features, bias=True)

        if self.softmax_for_final_decoder_layer:
            self.decoder_network_.add_module('softmax', Softmax(dim=1))

        self._offset_ = 0
        self.loss_ = 0
        self.optimizer_ = optim.Adam(
            list(self.encoder_network_.parameters()) + list(self.decoder_network_.parameters()),
            lr=self.learning_rate)

    def _optimize_params(self, inputs: torch.Tensor):
        self.optimizer_.zero_grad()

        reconstructed = self.decoder_network_(self.encoder_network_(inputs))
        self.loss_ = self.reconstruction_loss_function(inputs, reconstructed).mean()

        self.loss_.backward()
        self.optimizer_.step()

    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        mlflow.log_metrics(
            step=epoch,
            metrics=OrderedDict([
                ('Training time', epoch_train_time),
                ('Loss', self.loss_.item())]))
        print(f'Epoch {epoch}/{self.n_epochs},'
              f' Epoch training time: {epoch_train_time},'
              f' Loss: {self.loss_}')
