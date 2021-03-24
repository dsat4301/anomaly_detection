import math
from collections import OrderedDict
from typing import Tuple, Sequence, Callable

import mlflow
import numpy as np
import torch
from torch import optim
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from anomaly_detectors.GANomaly.GANomaly_loss import GANomalyLoss
from base.base_generative_anomaly_detector import BaseGenerativeAnomalyDetector
from base.base_networks import GeneratorNet, DiscriminatorNet


class GANomalyAnomalyDetector(BaseGenerativeAnomalyDetector):
    """ Semi-Supervised Anomaly Detection via Adversarial Training.

    Classification of samples as anomaly or normal data based on GANomaly architecture
    introduced by Akcay, S., Atapour-Abarghouei, A. & Breckon, T. P.
    Ganomaly: Semi-supervisedanomaly detection via adversarial trainingin
    Asian conference on computer vision (Springer, 2018), 622–637 (https://arxiv.org/pdf/1805.06725.pdf.).
    ----------

    Parameters
    ----------
    latent_dimensions : int, default=2
        Number of latent dimensions.
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
    device : {'cpu', 'cuda'}, default='cpu'
        Specifies the computational device using device agnostic code:
        (https://pytorch.org/docs/stable/notes/cuda.html).
    n_epochs : int, default=10
        Number of epochs.
    batch_size : int, default=128
        Batch size.
    n_jobs_dataloader : int, default=4
        Value for parameter num_workers of torch.utils.data.DataLoader
        (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
        Indicates how many subprocesses to use for data loading with values greater 0 enabling
        multi-process data loading.
    learning_rate : float, default=0.0001
        Learning rate.
    linear : bool, default=True
        Specifies if only linear layers without activation are used in encoder and decoder.
    n_hidden_features : Sequence[int], default=None
        Is Ignored if liner is True.
        Number of units used in the hidden encoder and decoder layers.
    random_state : int, default=None
        Seed value to be applied in order to create deterministic results.
    scorer : Callable
        Scorer instance to be used in score function.
    softmax_for_final_decoder_layer : bool, default=False
        Specifies if a softmax layer is inserted after the final decoder layer.
    reconstruction_loss_function : torch.nn.modules.loss._Loss, default=None
        The torch.nn.modules.loss._Loss instance for determining the reconstruction loss. If None, MSELoss is used.

    Attributes
    ----------
    generator_net_ : torch.nn.Module
        The generator network.
    discriminator_net_ : torch.nn.Module
        The discriminator network.

    Examples
    --------
    >>> import numpy
    >>> from anomaly_detectors.GANomaly.GANomaly_anomaly_detector import GANomalyAnomalyDetector
    >>> data = numpy.array([[0], [0.44], [0.45], [0.46], [1]])
    >>> ganomaly = GANomalyAnomalyDetector().fit(data)
    >>> ganomaly.score_samples(data)
    array([2.84801, 3.31392, 3.32494, 3.33598, 3.96101])
    """

    PRECISION = 5

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
        return [self.generator_net_, self.discriminator_net_]

    @property
    def _reset_loss_func(self) -> Callable:
        def reset_losses():
            self._training_loss_epoch_.reset()
            self._validation_loss_epoch_.reset()

        return reset_losses

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
            scorer: Callable = None,
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
            softmax_for_final_decoder_layer=softmax_for_final_decoder_layer,
            reconstruction_loss_function=reconstruction_loss_function)

        self.weight_adverserial_loss = weight_adverserial_loss
        self.weight_contextual_loss = weight_contextual_loss
        self.weight_encoder_loss = weight_encoder_loss
        self.optimizer_betas = optimizer_betas

        if self.reconstruction_loss_function is not None \
                and self.reconstruction_loss_function.reduction != 'none':
            raise ValueError('Invalid reduction for loss.')

    # noinspection PyPep8Naming
    def score_samples(self, X: np.ndarray):
        """ Return the anomaly score.

        :param X: numpy.ndarray of shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        :return: numpy.ndarray with shape (n_samples,)
            Array with positive scores with higher values indicating higher probability of the
            sample being an anomaly.
        """
        X, _ = self._check_ready_for_prediction(X)

        # noinspection PyTypeChecker
        loader = self._get_data_loader(X, shuffle=False)

        scores = []

        self.generator_net_.eval()

        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device=self.device)

                _, latent_input, latent_output = self.generator_net_(inputs)
                anomaly_scores = torch.mean(torch.pow(latent_input - latent_output, 2), dim=1)

                scores += anomaly_scores.cpu().data.numpy().tolist()

        return np.array(scores).round(self.PRECISION)

    def _initialize_fitting(self, train_loader: DataLoader):

        n_hidden_features_fallback = \
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

        self._optimizer_generator_ = optim.Adam(
            params=self.generator_net_.parameters(),
            lr=self.learning_rate,
            betas=self.optimizer_betas)
        self._optimizer_discriminator_ = optim.Adam(
            params=self.discriminator_net_.parameters(),
            lr=self.learning_rate,
            betas=self.optimizer_betas)
        self._training_loss_epoch_ = GANomalyLoss(
            device=self.device,
            weight_adverserial_loss=self.weight_adverserial_loss,
            weight_contextual_loss=self.weight_contextual_loss,
            weight_encoder_loss=self.weight_encoder_loss,
            reconstruction_loss_function=self.reconstruction_loss_function)
        self._validation_loss_epoch_ = GANomalyLoss(
            device=self.device,
            weight_adverserial_loss=self.weight_adverserial_loss,
            weight_contextual_loss=self.weight_contextual_loss,
            weight_encoder_loss=self.weight_encoder_loss,
            reconstruction_loss_function=self.reconstruction_loss_function)

        self._offset_ = 0

    def _optimize_params(self, inputs: torch.Tensor):

        generator_output = self.generator_net_(inputs)
        classifier_real, features_real = self.discriminator_net_(inputs)
        classifier_fake, features_fake = self.discriminator_net_(generator_output[0].detach())

        generator_loss = self._training_loss_epoch_.update_generator_loss(
            inputs,
            generator_output,
            features_real,
            features_fake)

        self._optimizer_generator_.zero_grad()
        generator_loss.backward(retain_graph=True)
        self._optimizer_generator_.step()

        discriminator_loss = self._training_loss_epoch_.update_discriminator_loss(classifier_real, classifier_fake)

        self._optimizer_discriminator_.zero_grad()
        discriminator_loss.backward()
        self._optimizer_discriminator_.step()

    def _log_epoch_results(self, epoch: int, epoch_train_time: float):
        metrics = OrderedDict([
            ('Training time', epoch_train_time),
            ('Training adverserial loss', self._training_loss_epoch_.adverserial_loss),
            ('Training contextual loss', self._training_loss_epoch_.contextual_loss),
            ('Training encoder loss', self._training_loss_epoch_.encoder_loss),
            ('Training generator loss', self._training_loss_epoch_.generator_loss),
            ('Training discriminator loss', self._training_loss_epoch_.discriminator_loss)])

        if self._validation_loss_epoch_.generator_loss \
                and self._validation_loss_epoch_.discriminator_loss \
                and self._validation_loss_epoch_.adverserial_loss \
                and self._validation_loss_epoch_.contextual_loss \
                and self._validation_loss_epoch_.encoder_loss:
            metrics['Validation generator loss'] = self._validation_loss_epoch_.generator_loss
            metrics['Validation discriminator loss'] = self._validation_loss_epoch_.discriminator_loss
            metrics['Validation adverserial loss'] = self._validation_loss_epoch_.adverserial_loss
            metrics['Validation contextual loss'] = self._validation_loss_epoch_.contextual_loss
            metrics['Validation encoder loss'] = self._validation_loss_epoch_.encoder_loss

        mlflow.log_metrics(step=epoch, metrics=metrics)

        print(f'Epoch {epoch}/{self.n_epochs},'
              f' Epoch training time: {epoch_train_time},'
              f" Generator Loss: {self._training_loss_epoch_.generator_loss},"
              f" Discriminator Loss: {self._training_loss_epoch_.discriminator_loss}")

    def _update_validation_loss_epoch(self, epoch: int, inputs: torch.Tensor):
        generator_output = self.generator_net_(inputs)
        classifier_real, features_real = self.discriminator_net_(inputs)
        classifier_fake, features_fake = self.discriminator_net_(generator_output[0].detach())

        _ = self._validation_loss_epoch_.update_generator_loss(
            inputs,
            generator_output,
            features_real,
            features_fake)

        _ = self._validation_loss_epoch_.update_discriminator_loss(classifier_real, classifier_fake)
