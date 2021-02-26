from abc import abstractmethod
from typing import Callable, Sequence

import numpy as np
import torch
from torch.nn import MSELoss
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from base.base_nn_anomaly_detector import BaseNNAnomalyDetector


class BaseGenerativeAnomalyDetector(BaseNNAnomalyDetector):
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
            latent_dimensions: int,
            softmax_for_final_decoder_layer: bool,
            reconstruction_loss_function: _Loss):
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
            novelty,
            latent_dimensions)

        self.softmax_for_final_decoder_layer = softmax_for_final_decoder_layer
        self.reconstruction_loss_function = reconstruction_loss_function

    @property
    @abstractmethod
    def offset_(self):
        raise NotImplemented

    @property
    @abstractmethod
    def _networks(self) -> Sequence[torch.nn.Module]:
        raise NotImplemented

    @property
    @abstractmethod
    def _reset_loss_func(self) -> Callable:
        raise NotImplemented

    # noinspection PyPep8Naming
    @abstractmethod
    def score_samples(self, X: np.ndarray):
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
