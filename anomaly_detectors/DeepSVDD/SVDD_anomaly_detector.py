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
    """ Deep One-Class Classification.

    Classification of samples as anomaly or normal data based on Deep SVDD architecture,
    introduced by Ruff, L.et al. Deep one-class classification
    in International conference on machine learning (2018), 4393–4402
    (http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf).

    Parameters
    ----------
    optimizer_name : str, default='adam'
        The name of the optimizer.
    learning_rate : float, default=1e-4
        Learning rate.
    n_epochs : int, default=10
        The number of epochs.
    batch_size : int, default=128
        Batch size.
    weight_decay : float, default=1e-6
        The value for weight decay regularization, applied during optimization.
    device : {'cpu', 'cuda'}, default='cpu'
        Specifies the computational device using device agnostic code:
        (https://pytorch.org/docs/stable/notes/cuda.html).
    n_jobs_dataloader : int, default=4
        Value for parameter num_workers of torch.utils.data.DataLoader
        (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
        Indicates how many subprocesses to use for data loading with values greater 0 enabling
        multi-process data loading.
    latent_dimensions : int, default=2
        Number of latent dimensions.
    linear : bool, default=True
        Specifies if only linear layers without activation are used in encoder and decoder.
    n_hidden_features : Sequence[int], default=None
        Is Ignored if liner is True.
        Number of units used in the hidden encoder and decoder layers.
    random_state : int, default=None
        Seed value to be applied in order to create deterministic results.
    scorer : Callable
        Scorer instance to be used in score function.

    Attributes
    ----------
    network_ : torch.nn.Module
        The network.
    hypersphere_center_ : torch.Tensor
        The center of the hypersphere, learned during training.

    Examples
    --------
    >>> import numpy
    >>> from anomaly_detectors.DeepSVDD.SVDD_anomaly_detector import DeepSVDDAnomalyDetector
    >>> data = numpy.array([[0], [0.44], [0.45], [0.46], [1]])
    >>> deep_svdd = DeepSVDDAnomalyDetector().fit(data)
    >>> deep_svdd.score_samples(data)
    array([2.94518113, 1.94397306, 1.92362654, 1.90338707, 0.9694134 ])
    """

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
        """ Gets the threshold, applied for decision_function.
        :rtype : float
        """
        return self._offset_

    @offset_.setter
    def offset_(self, value: float):
        """ Sets the threshold, applied for decision_function.
        :param value : float
        """
        # noinspection PyAttributeOutsideInit
        self._offset_ = value

    @property
    def _networks(self) -> Sequence[torch.nn.Module]:
        return [self.network_]

    @property
    def _reset_loss_func(self) -> Callable:
        def reset_loss():
            self._loss_epoch_ = []
            self._validation_loss_epoch_ = []

        return reset_loss

    # noinspection PyPep8Naming,SpellCheckingInspection
    def score_samples(self, X):
        """ Return the anomaly score.

        :param X : numpy.ndarray of shape (n_samples, n_features)
            Set of samples to be scored, where n_samples is the number of samples and
            n_features is the number of features.

        :return : numpy.ndarray with shape (n_samples,)
            Array with positive scores.
            Higher values indicate that an instance is more likely to be anomalous.
        """
        X, _ = self._check_ready_for_prediction(X)

        # noinspection PyTypeChecker
        loader = self._get_data_loader(X, shuffle=False)

        scores = []
        self.network_.eval()

        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device=self.device)
                outputs = self.network_(inputs)

                anomaly_scores = self._get_distances_to_center(outputs, reduction='none')
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
        self._optimizer_ = optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.optimizer_name == 'amsgrad')

        self.hypersphere_center_ = self._get_initial_center_c(train_loader)
        self._loss_epoch_ = None
        self._validation_loss_epoch_ = None
        self._offset_ = 0

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
        outputs = self.network_(inputs)
        loss = self._get_distances_to_center(outputs, reduction='none')

        self._optimizer_.zero_grad()
        loss.mean().backward()
        self._optimizer_.step()

        self._loss_epoch_ += loss.data.numpy().tolist()

    def _get_distances_to_center(self, outputs: torch.Tensor, reduction: str):
        distances = torch.sum((outputs - self.hypersphere_center_) ** 2, dim=1)
        return distances if reduction == 'none' else distances.mean()

    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        mean_training_loss_epoch = np.array(self._loss_epoch_).mean()

        metrics = OrderedDict([
            ('Training time', epoch_train_time),
            ('Training Loss', mean_training_loss_epoch)])

        if self._validation_loss_epoch_:
            metrics['Validation Loss'] = np.array(self._validation_loss_epoch_).mean()

        mlflow.log_metrics(step=epoch, metrics=metrics)
        print(f'Epoch {epoch}/{self.n_epochs},'
              f' Epoch training time: {epoch_train_time},'
              f' Loss: {mean_training_loss_epoch}')

    def _update_validation_loss_epoch(self, epoch: int, inputs: torch.Tensor):
        outputs = self.network_(inputs)
        loss = self._get_distances_to_center(outputs, reduction='none')

        self._validation_loss_epoch_ += loss.data.numpy().tolist()
