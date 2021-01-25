import math
from collections import OrderedDict
from typing import Tuple, Sequence, Callable

import mlflow
import numpy as np
import torch
from sklearn.metrics import make_scorer, roc_auc_score
from torch import optim
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from anomaly_detectors.GANomaly.GANomaly_loss import GANomalyLoss
from base.base_anomaly_detector import BaseAnomalyDetector
from base.base_networks import GeneratorNet, DiscriminatorNet


class GANomalyAnomalyDetector(BaseAnomalyDetector):
    """Semi-Supervised Anomaly Detection via Adversarial Training.

    Classification of samples as anomaly or normal data based on GANomaly architecture
    introduced by: https://arxiv.org/pdf/1805.06725.pdf.
    ----------
    size_z : int, default=2
        Latent space dimensions equivalent to the number of neurons in the last encoder layer.
    weight_adverserial_loss : float, default=1
        Weight of the adverserial loss term in the generator loss-function.
    weight_contextual_loss : float, default=50
        Weight of the contextual loss term in the generator loss-function.
    weight_encoder_loss : float, default=1
        Weight of the encoder loss term in the generator loss-function.
    optimizer_betas : Tuple[float, float], default=(0.5, 0.999)
        Value for parameter betas of torch optimizers
        (https://pytorch.org/docs/stable/optim.html#module-torch.optim).
        Indicates the coefficients used for computing running averages of gradient and its square.

    Examples
    --------
    >>> from anomaly_detectors.GANomaly.GANomaly_anomaly_detector import GANomalyAnomalyDetector
    >>> data = np.array([[0], [0.44], [0.45], [0.46], [1]])
    >>> ganomaly = GANomalyAnomalyDetector().fit(data)
    >>> ganomaly.score_samples(data)
    array([0.35026681, 0.53358972, 0.53837019, 0.54317802, 0.84332526])
    >>> ganomaly.predict(data)
    array([0, 1, 1, 1, 1])
    """
    @property
    def offset_(self):
        return self._offset_

    @offset_.setter
    def offset_(self, value: float):
        # noinspection PyAttributeOutsideInit
        self._offset_ = value

    @property
    def _networks(self) -> Sequence[torch.nn.Module]:
        return [self.generator_net_, self.discriminator_net_]

    @property
    def _reset_loss_func(self) -> Callable:
        return self.loss_.reset

    def __init__(
            self,
            device: str = 'cpu',
            n_epochs: int = 10,
            batch_size: int = 256,
            n_jobs_dataloader: int = 4,
            learning_rate: float = 0.0001,
            latent_dimensions: int = 2,
            weight_adverserial_loss: float = 1,
            weight_contextual_loss: float = 50,
            weight_encoder_loss=1,
            optimizer_betas: Tuple[float, float] = (0.5, 0.999),
            linear: bool = True,
            n_hidden_features: Sequence[int] = None,
            random_state: int = None,
            scorer: Callable = make_scorer(roc_auc_score, needs_threshold=True),
            softmax_for_final_decoder_layer: bool = False):
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

        self.weight_adverserial_loss = weight_adverserial_loss
        self.weight_contextual_loss = weight_contextual_loss
        self.weight_encoder_loss = weight_encoder_loss
        self.optimizer_betas = optimizer_betas
        self.softmax_for_final_decoder_layer = softmax_for_final_decoder_layer

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        """ Return the anomaly score.

        :param X: np.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :return: np.ndarray with shape (n_samples,)
            Array with positive scores with higher values indicating higher probability of the
            sample being an anomaly.
        """
        X, _ = self._check_ready_for_prediction(X)

        # noinspection PyTypeChecker
        loader = self._get_test_loader(X)

        scores = []

        self.generator_net_.eval()

        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device=self.device)

                _, latent_input, latent_output = self.generator_net_(inputs)
                anomaly_scores = torch.mean(torch.pow(latent_input - latent_output, 2), dim=1)

                scores += anomaly_scores.cpu().data.numpy().tolist()

        return np.array(scores)

    def _initialize_fitting(self, train_loader: DataLoader):
        print(self.n_features_in_)
        print(self.latent_dimensions)

        n_hidden_features_fallback =\
            [self.n_features_in_ - math.floor((self.n_features_in_ - self.latent_dimensions) / 2)]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.generator_net_ = GeneratorNet(
            self.latent_dimensions,
            self.n_features_in_,
            self.n_hidden_features if self.n_hidden_features is not None else n_hidden_features_fallback,
            self.linear).to(self.device)
        self.discriminator_net_ = DiscriminatorNet(
            self.latent_dimensions,
            self.n_features_in_,
            self.n_hidden_features if self.n_hidden_features is not None else n_hidden_features_fallback,
            self.linear).to(self.device)

        if self.softmax_for_final_decoder_layer:
            self.generator_net_.decoder.add_module('softmax', torch.nn.Softmax(dim=1))

        self.optimizer_generator_ = optim.Adam(
            params=self.generator_net_.parameters(),
            lr=self.learning_rate,
            betas=self.optimizer_betas)
        self.optimizer_discriminator_ = optim.Adam(
            params=self.discriminator_net_.parameters(),
            lr=self.learning_rate,
            betas=self.optimizer_betas)
        self.loss_ = GANomalyLoss(
            device=self.device,
            weight_adverserial_loss=self.weight_adverserial_loss,
            weight_contextual_loss=self.weight_contextual_loss,
            weight_encoder_loss=self.weight_encoder_loss)
        self._offset_ = 0

    def _optimize_params(self, inputs: torch.Tensor):
        self.optimizer_generator_.zero_grad()
        self.optimizer_discriminator_.zero_grad()

        # Forward pass
        generator_output = self.generator_net_(inputs)
        classifier_real, features_real = self.discriminator_net_(inputs)
        classifier_fake, features_fake = self.discriminator_net_(generator_output[0].detach())

        # Backward pass

        # Generator
        generator_loss = self.loss_.update_generator_loss(inputs,
                                                          generator_output,
                                                          features_real,
                                                          features_fake)
        generator_loss.backward(retain_graph=True)
        self.optimizer_generator_.step()

        # Discriminator
        discriminator_loss = self.loss_.update_discriminator_loss(classifier_real, classifier_fake)
        discriminator_loss.backward()
        self.optimizer_discriminator_.step()

    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        mlflow.log_metrics(
            step=epoch,
            metrics=OrderedDict([
                ('Training time', epoch_train_time),
                ('Adverserial loss', self.loss_.adverserial_loss_epoch.item()),
                ('Contextual loss', self.loss_.contextual_loss_epoch.item()),
                ('EncoderLinear loss', self.loss_.encoder_loss_epoch.item()),
                ('Generator loss', self.loss_.generator_loss_epoch.item()),
                ('Discriminator loss', self.loss_.discriminator_loss_epoch.item())]))
        print(f'Epoch {epoch}/{self.n_epochs},'
              f' Epoch training time: {epoch_train_time},'
              f' Generator Loss: {self.loss_.generator_loss_epoch},'
              f' Discriminator Loss: {self.loss_.discriminator_loss_epoch}')
