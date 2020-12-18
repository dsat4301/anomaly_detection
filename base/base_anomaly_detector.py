import time
from abc import abstractmethod
from typing import Sequence, Callable

import mlflow
import numpy as np
import torch
from base_dataset import BaseDataset
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.utils import check_X_y, check_array
# noinspection PyProtectedMember
from torch.utils.data import DataLoader


class BaseAnomalyDetector(BaseEstimator, OutlierMixin):

    def __init__(
            self,
            batch_size: int = 128,
            n_jobs_dataloader: int = 0,
            n_epochs: int = 10,
            device: str = 'cpu',
            threshold: int = .5,
            scorer: Callable = make_scorer(average_precision_score, needs_threshold=True)):

        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.device = device
        self.threshold = threshold
        self.scorer = scorer

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
        """ Trains generator and discriminator based on the normal samples in X.

        :param X : np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :param y : binary np.ndarray of with shape (n_samples,), default=None
            If given, y is used to filter normal values from X. This means only the samples of X
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

        if y is not None:
            X, y = check_X_y(X, y, estimator=self)
            if len(np.unique(y)) > 2:
                raise ValueError

            normal_data = X[y == min(y)]
            if len(normal_data) == 0:
                print(y)
                raise ValueError
        else:
            normal_data = check_array(X)

        # noinspection PyAttributeOutsideInit
        self.n_features_in_ = np.array(X).shape[1]

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

    # noinspection PyPep8Naming
    @abstractmethod
    def decision_function(self, X: np.ndarray):
        raise NotImplemented

    # noinspection PyPep8Naming
    def score(self, X: np.ndarray, y: np.ndarray):
        """ Return the score based on the scorer passed as class parameter.
        :param X : np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :param y : np.ndarray of with shape (n_samples,), default=None
            The true anomaly labels.
        :return : float
            Scalar score value.
        """
        X, y = check_X_y(X, y)
        return self.scorer(estimator=self, X=X, y_true=y)

    # noinspection PyPep8Naming
    def predict(self, X):
        """ Return the classification result based on the results of decision_function.
        :param X: np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :return: np.ndarray with shape (n_samples,)
            Binary array with -1 indicating an anomaly and 1 indicating normal data.
        """

        anomaly_scores = self.decision_function(X)
        v_mapping = np.vectorize(lambda x: -1 if x else 1)

        return v_mapping((anomaly_scores >= self.threshold))

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

    @staticmethod
    def get_mapped_prediction(y: np.ndarray):
        return np.vectorize(lambda x: 0 if x == 1 else 1)(y)

    def _more_tags(self):
        return {'binary_only': True}
