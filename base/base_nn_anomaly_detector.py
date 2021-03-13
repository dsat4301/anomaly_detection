import time
from abc import abstractmethod
from typing import Sequence, Callable

import mlflow
import numpy as np
import torch
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from base.base_anomaly_detector import BaseAnomalyDetector
from util.data import BaseDataset


class BaseNNAnomalyDetector(BaseAnomalyDetector):
    """ Base anomaly detector class implementing scikit-learn's BaseEstimator and OutlierMixin

    batch_size : int, default=256
        Batch size.
    n_jobs_dataloader : int, default=4
        Value for parameter num_workers of torch.utils.data.DataLoader
        (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
        Indicates how many subprocesses to use for data loading with values greater 0 enabling
        multi-process data loading.
    n_epochs : int, default=10
        Number of epochs.
    device : {'cpu', 'cuda'}, default='cpu'
        Specifies the computational device using device agnostic code:
        (https://pytorch.org/docs/stable/notes/cuda.html).
    threshold : float = .5
        Threshold to be used for predict function.
        Values greater than or equal to threshold will be classified as anomalies.
    scorer : Callable, default=make_scorer(roc_auc_score, needs_threshold=True)
        Scorer instance to be used in score function.
    learning_rate : int, default=0.0001
        Learning rate.
    linear : bool, default=True
        Specifies if only linear layers without activation should be used in the subnetworks.
    n_hidden_features : Sequence[int], default=None
        Is Ignored if liner is True.
        Determines the number of neurons to be used in the subnetwork layers.
        Expects a Sequence with decreasingly ordered values with the last value greater than size_z.
    random_state : int, default=None
        Value for torch.seed. If None, no seed will be used.
    """

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
        :param y : binary np.ndarray of with shape (n_samples,), default=None
            If given, y is used to filter normal values from data. This means only the samples of data
            with the smaller label in y are used for training.
        :param kwargs :
            is_logging_enabled: bool, default=False
                Indicates if epoch results should be printed to the console resp. to a mlflow
                run which has to be started outside.
        :return : GANomalyEstimator
            The fitted instance.
        """

        is_logging_enabled = False
        if 'is_logging_enabled' in kwargs and kwargs['is_logging_enabled'] is True:
            is_logging_enabled = True
            mlflow.log_params(self.get_params())

        normal_data = self._get_normal_data(X, y)
        self._set_n_features_in(normal_data)

        train_set = BaseDataset(data=normal_data)
        train_loader = DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.n_jobs_dataloader)

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
                self._log_epoch_results(epoch + 1, epoch_train_time)

        train_time = time.time() - start_time

        if is_logging_enabled:
            mlflow.log_param('Train time', train_time)
            print('Finished training...')

        return self

    def _get_test_loader(self, data: np.ndarray):
        prediction_set = BaseDataset(data=data)
        prediction_loader = DataLoader(
            dataset=prediction_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs_dataloader)
        return prediction_loader

    @abstractmethod
    def _initialize_fitting(self, train_loader: DataLoader):
        raise NotImplemented

    @abstractmethod
    def _optimize_params(self, inputs: torch.Tensor):
        raise NotImplemented

    @abstractmethod
    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        raise NotImplemented
