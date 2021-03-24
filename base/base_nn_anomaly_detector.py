import time
from abc import abstractmethod
from typing import Sequence, Callable

import mlflow
import numpy as np
import torch
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from base.base_anomaly_detector import BaseAnomalyDetector
from base.base_dataset import BaseDataset


class BaseNNAnomalyDetector(BaseAnomalyDetector):
    """Base class for neural network-based anomaly detectors, implementing BaseAnomalyDetector."""

    def __init__(
            self,
            batch_size: int,
            n_jobs_dataloader: int,
            n_epochs: int,
            device: str,
            scorer: Callable,
            learning_rate: float,
            linear: bool,
            n_hidden_features: Sequence[int],
            random_state: int,
            novelty: bool,
            latent_dimensions: int):
        super(BaseNNAnomalyDetector, self).__init__(scorer, random_state)

        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.linear = linear
        self.n_hidden_features = n_hidden_features
        self.novelty = novelty
        self.latent_dimensions = latent_dimensions

    @property
    @abstractmethod
    def _networks(self) -> Sequence[torch.nn.Module]:
        raise NotImplemented

    @property
    @abstractmethod
    def _reset_loss_func(self) -> Callable:
        raise NotImplemented

    # noinspection PyPep8Naming,PyAttributeOutsideInit
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """ Trains generator and discriminator based on the normal samples in data.

        :param X : np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :param y : binary np.ndarray of shape (n_samples,), default=None
            If given, y is used to filter normal values from data. This means only the samples of data
            with the smaller label in y are used for training.
        :param kwargs :
            is_logging_enabled: bool, default=False
                Indicates if epoch results should be printed to the console resp. to a mlflow
                run which has to be started outside.
            X_validation : np.ndarray of shape (n_samples, n_features)
                Set of validation samples. It must have the same number of features like X.
                The set is ignored, if the logging is disabled.
            y_validation : binary np.ndarray of shape (n_samples,)
                The validation targets, used to filter normal values from the validation set
                and to log the scores using the score function for each epoch.
                It is ignored if logging is disabled (default).
        :return : BaseNNAnomalyDetector
            The fitted instance.
        """

        normal_data = self._get_normal_data(X, y)
        self._set_n_features_in(normal_data)
        train_loader = self._get_data_loader(data=normal_data, shuffle=True)

        is_logging_enabled = kwargs.get('is_logging_enabled', False)
        X_validation = kwargs.get('X_validation')
        y_validation = kwargs.get('y_validation')
        validation_loader = None

        if is_logging_enabled:
            mlflow.log_params(self.get_params())

            if X_validation is not None:
                normal_validation_data = self._get_normal_data(X_validation, y_validation)
                validation_loader = self._get_data_loader(data=normal_validation_data, shuffle=False)

        self._initialize_fitting(train_loader)

        for network in self._networks:
            network.train()

        if is_logging_enabled:
            print('\nStarting training...')
        start_time = time.time()

        for epoch in range(self.n_epochs):

            epoch_start_time = time.time()
            self._reset_loss_func()

            for inputs in train_loader:
                inputs = inputs.to(device=self.device)
                self._optimize_params(inputs)

            epoch_train_time = time.time() - epoch_start_time

            if is_logging_enabled:
                if validation_loader is not None:
                    with torch.no_grad():
                        for inputs in validation_loader:
                            inputs = inputs.to(self.device)
                            self._update_validation_loss_epoch(epoch + 1, inputs)

                if X_validation is not None and y_validation is not None:
                    mlflow.log_metric('Score train', self.score(X, y), step=epoch + 1)
                    mlflow.log_metric('Score validation', self.score(X_validation, y_validation), step=epoch + 1)

                self._log_epoch_results(epoch + 1, epoch_train_time)

        train_time = time.time() - start_time

        if is_logging_enabled:
            mlflow.log_param('Train time', train_time)
            print('Finished training...')

        return self

    def _get_data_loader(self, data: np.ndarray, shuffle: bool):
        return DataLoader(
            BaseDataset(data),
            batch_size=self.batch_size,
            num_workers=self.n_jobs_dataloader,
            shuffle=shuffle)

    @abstractmethod
    def _update_validation_loss_epoch(self, epoch: int, inputs: torch.Tensor):
        raise NotImplemented

    @abstractmethod
    def _initialize_fitting(self, train_loader: DataLoader):
        raise NotImplemented

    @abstractmethod
    def _optimize_params(self, inputs: torch.Tensor):
        raise NotImplemented

    @abstractmethod
    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        raise NotImplemented
