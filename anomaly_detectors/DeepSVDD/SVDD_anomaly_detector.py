from collections import OrderedDict
from typing import Callable, Sequence

import mlflow
import numpy as np
import torch
from torch import optim, nn
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from base.base_networks import Encoder
from base.base_nn_anomaly_detector import BaseNNAnomalyDetector


class DeepSVDDAnomalyDetector(BaseNNAnomalyDetector):

    def __init__(
            self,
            optimizer_name: str = 'adam',
            learning_rate: float = .0001,
            n_epochs: int = 10,
            batch_size: int = 128,
            weight_decay: float = 1e-6,
            device: str = 'cpu',
            n_jobs_dataloader: int = 4,
            latent_dimensions: int = 2,
            linear: bool = True,
            n_hidden_features: Sequence[int] = None,
            random_state: int = None,
            scorer: Callable = None):

        super().__init__(
            batch_size=batch_size,
            n_jobs_dataloader=n_jobs_dataloader,
            n_epochs=n_epochs,
            device=device,
            scorer=scorer,
            learning_rate=learning_rate,
            linear=linear,
            n_hidden_features=n_hidden_features,
            random_state=random_state,
            novelty=True,
            latent_dimensions=latent_dimensions)

        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay

    @property
    def offset_(self):
        return self.__offset_

    @offset_.setter
    def offset_(self, value: float):
        # noinspection PyAttributeOutsideInit
        self.__offset_ = value

    @property
    def _networks(self) -> Sequence[torch.nn.Module]:
        return [self.network_]

    @property
    def _reset_loss_func(self) -> Callable:
        def reset_loss():
            pass

        return reset_loss

    # noinspection PyPep8Naming,SpellCheckingInspection
    def score_samples(self, X):
        """ Return the anomaly score.
        :param X: np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :return: np.ndarray with shape (n_samples,)
            Array with positive scores with higher values indicating higher probability of the
            sample beeing an anomaly.
        """
        X, _ = self._check_ready_for_prediction(X)

        # noinspection PyTypeChecker
        loader = self._get_test_loader(X)

        scores = []
        self.network_.eval()

        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device=self.device)
                output = self.network_(inputs)
                dist = torch.sum((output - self.c_) ** 2, dim=1)

                anomaly_scores = dist
                scores += anomaly_scores.cpu().data.numpy().tolist()

        return np.array(scores)

    def _initialize_fitting(self, train_loader: DataLoader):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.network_ = nn.Sequential(nn.Linear(self.n_features_in_, self.latent_dimensions, bias=False)) \
            if self.linear \
            else Encoder(self.latent_dimensions, self.n_features_in_, self.n_hidden_features, bias=False)

        self.network_.to(self.device)

        # noinspection SpellCheckingInspection
        self.optimizer_ = optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.optimizer_name == 'amsgrad')

        self.c_ = self._get_initial_center_c(train_loader)
        self.loss_ = 0
        self.__offset_ = 0

    def _get_initial_center_c(self, train_loader: DataLoader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        c = torch.zeros(self.latent_dimensions, device=self.device)

        self.network_.eval()
        with torch.no_grad():
            for inputs in train_loader:
                # get the inputs of the batch
                inputs = inputs.to(self.device)
                outputs = self.network_(inputs)

                c += torch.sum(outputs, dim=0)

        c /= len(train_loader)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def _optimize_params(self, inputs: torch.Tensor):

        # Zero the network parameter gradients
        self.optimizer_.zero_grad()

        # Update network parameters via backpropagation: forward + backward + optimize
        outputs = self.network_(inputs)
        dist = torch.sum((outputs - self.c_) ** 2, dim=1)

        self.loss_ = torch.mean(dist)

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
