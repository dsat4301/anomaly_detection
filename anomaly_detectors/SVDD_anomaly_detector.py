from collections import OrderedDict
from typing import Callable, Sequence

import mlflow
import torch
import numpy as np
from base_networks import EncoderLinear, Encoder
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.utils.validation import check_is_fitted, check_array
from torch import optim
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from base.base_anomaly_detector import BaseAnomalyDetector


class SVDDAnomalyDetector(BaseAnomalyDetector):

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
            threshold: int = .5,
            scorer: Callable = make_scorer(average_precision_score, needs_threshold=True)):

        super().__init__(
            batch_size=batch_size,
            n_jobs_dataloader=n_jobs_dataloader,
            n_epochs=n_epochs,
            device=device,
            threshold=threshold,
            scorer=scorer)

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.latent_dimensions = latent_dimensions
        self.linear = linear
        self.n_hidden_features = n_hidden_features
        self.random_state = random_state

    @property
    def _networks(self) -> Sequence[torch.nn.Module]:
        return [self.network_]

    @property
    def _reset_loss_func(self) -> Callable:
        def reset_loss():
            pass

        return reset_loss

    # noinspection PyPep8Naming,SpellCheckingInspection
    def decision_function(self, X):
        """ Return the anomaly score.
        :param X: np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :return: np.ndarray with shape (n_samples,)
            Array with positive scores with higher values indicating higher probability of the
            sample beeing an anomaly.
        """
        check_is_fitted(self)
        X = check_array(X, estimator=self)
        # noinspection PyUnresolvedReferences
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid number of features in data.')

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

        self.network_ = EncoderLinear(self.latent_dimensions, self.n_features_in_) \
            if self.linear \
            else Encoder(self.latent_dimensions, self.n_features_in_, self.n_hidden_features)

        self.network_.to(self.device)

        # noinspection SpellCheckingInspection
        self.optimizer_ = optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.optimizer_name == 'amsgrad')

        self.c_ = self._get_initial_center_c(train_loader)
        self.loss_ = 0

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
