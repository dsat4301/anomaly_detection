from collections import OrderedDict
from typing import Callable, Sequence

import mlflow
import numpy as np
import torch
from torch import nn, optim
from torch.nn import Softmax
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from base.base_generative_anomaly_detector import BaseGenerativeAnomalyDetector
from base.base_networks import Encoder, Decoder


class AEAnomalyDetector(BaseGenerativeAnomalyDetector):
    """ Autoencoder-based anomaly detection.

    Prediction of anomaly scores for samples based on a reconstruction loss value.

    Parameters
    ----------
    batch_size : int, default=128
        Batch size.
    n_jobs_dataloader: int, default=4,
        Value for parameter num_workers of torch.utils.data.DataLoader
        (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
        Indicates how many subprocesses to use for data loading with values greater 0 enabling
        multi-process data loading.
    n_epochs: int, default=10,
        Number of epochs.
    device : {'cpu', 'cuda'}, default='cpu'
        Specifies the computational device using device agnostic code:
        (https://pytorch.org/docs/stable/notes/cuda.html).
    scorer : Callable
        Scorer instance to be used in score function.
    learning_rate: float, default=0.0001,
        Learning rate.
    linear : bool, default=True
        Specifies if only linear layers without activation are used in encoder and decoder.
    n_hidden_features : Sequence[int], default=None
        Is Ignored if liner is True.
        Number of units used in the hidden encoder and decoder layers.
    random_state: int, default=None
        Scorer instance used in score function.
    latent_dimensions: int, default=2
        Number of latent dimensions.
    softmax_for_final_decoder_layer: bool, default=False
        Specifies if a softmax layer is inserted after the final decoder layer.
    reconstruction_loss_function: _Loss, default=None
        The _Loss instance for determining the reconstruction loss. If None, MSELoss is used.

    Attributes
    ----------
    encoder_network_ : torch.nn.Module
        The encoder network.
    decoder_network_ : torch.nn.Module
        The decoder_network.

    Examples
    --------
    >>> from anomaly_detectors.autoencoder.AE_anomaly_detector import AEAnomalyDetector
    >>> data = np.array([[0], [0.44], [0.45], [0.46], [1]])
    >>> ae = AEAnomalyDetector().fit(data)
    >>> ae.score_samples(data)
    array([0.26844, 0.00374, 0.00258, 0.00163, 0.27086])
    """

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
            softmax_for_final_decoder_layer: bool = False,
            reconstruction_loss_function: _Loss = None):
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
            latent_dimensions=latent_dimensions,
            reconstruction_loss_function=reconstruction_loss_function,
            softmax_for_final_decoder_layer=softmax_for_final_decoder_layer)

        self.softmax_for_final_decoder_layer = softmax_for_final_decoder_layer

        if self.reconstruction_loss_function is not None \
                and self.reconstruction_loss_function.reduction != 'none':
            raise ValueError('Loss with reduction none required.')

    @property
    def offset_(self):
        """Gets the threshold, applied for decision_function."""
        return self._offset_

    @offset_.setter
    def offset_(self, value):
        """Sets the threshold, applied for decision_function"""
        # noinspection PyAttributeOutsideInit
        self._offset_ = value

    @property
    def _networks(self) -> Sequence[torch.nn.Module]:
        return [self.encoder_network_, self.decoder_network_]

    @property
    def _reset_loss_func(self) -> Callable:
        def func():
            self._loss_epoch_ = []
            self._validation_loss_epoch_ = []

        return func

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        """Return the anomaly score.

        :param X: numpy.ndarray of shape (n_samples, n_features)
            Set of samples to be scored, where n_samples is the number of samples and
            n_features is the number of features.

        :return: numpy.ndarray with shape (n_samples,)
            Array with positive scores.
            Higher values indicate that an instance is more likely to be anomalous.
        """
        X, _ = self._check_ready_for_prediction(X)

        # noinspection PyTypeChecker
        loader = self._get_data_loader(data=X, shuffle=False)

        scores = []
        self.encoder_network_.eval()
        self.decoder_network_.eval()

        with torch.no_grad():
            for inputs in loader:
                reconstructed = self.decoder_network_(self.encoder_network_(inputs))

                anomaly_scores = self._get_loss_function_value(inputs, reconstructed)

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
        self._loss_epoch_ = None
        self._validation_loss_epoch_ = None
        self._optimizer_ = optim.Adam(
            list(self.encoder_network_.parameters()) + list(self.decoder_network_.parameters()),
            lr=self.learning_rate)

    def _optimize_params(self, inputs: torch.Tensor):
        reconstructed = self.decoder_network_(self.encoder_network_(inputs))
        current_loss = self._get_loss_function_value(inputs, reconstructed)

        # Backpropagation
        self._optimizer_.zero_grad()
        current_loss.mean().backward()
        self._optimizer_.step()

        self._loss_epoch_ += current_loss.data.numpy().tolist()

    def _get_loss_function_value(
            self,
            inputs: torch.Tensor,
            reconstructed: torch.Tensor) -> torch.Tensor:

        loss = self.reconstruction_loss_function(inputs, reconstructed) \
            if self.reconstruction_loss_function is not None \
            else nn.MSELoss(reduction='none')(inputs, reconstructed)

        return loss.mean(axis=1)

    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        mean_training_loss_epoch = np.array(self._loss_epoch_).mean()

        metrics = OrderedDict([
            ('Training time', epoch_train_time),
            ('Training Loss', mean_training_loss_epoch)])

        if self._validation_loss_epoch_:
            metrics['Validation loss'] = np.array(self._validation_loss_epoch_).mean()

        mlflow.log_metrics(step=epoch, metrics=metrics)
        print(f'Epoch {epoch}/{self.n_epochs},'
              f' Epoch training time: {epoch_train_time},'
              f' Loss: {mean_training_loss_epoch}')

    def _update_validation_loss_epoch(self, epoch: int, inputs: torch.Tensor):
        reconstructed = self.decoder_network_(self.encoder_network_(inputs))
        loss = self._get_loss_function_value(inputs, reconstructed)

        self._validation_loss_epoch_ += loss.data.numpy().tolist()
