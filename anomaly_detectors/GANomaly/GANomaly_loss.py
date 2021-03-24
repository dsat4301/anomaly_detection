import numpy as np
import torch
from torch import nn
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss, L1Loss


class GANomalyLoss:
    """ Class for collecting information regarding loss values during training of a GANomaly network.

    Parameters
    ----------
    device : {'cpu', 'cuda'}, default='cpu'
        Specifies the computational device using device agnostic code:
        (https://pytorch.org/docs/stable/notes/cuda.html).
    weight_adverserial_loss : float, default=1
        Weight of the adverserial loss term in the generator loss-function.
    weight_contextual_loss : float, default=50
        Weight of the contextual loss term in the generator loss-function.
    weight_encoder_loss : float, default=1
        Weight of the encoder loss term in the generator loss-function.
    reconstruction_loss_function : torch.nn.modules.loss._Loss, default=None
        The torch.nn.modules.loss._Loss instance for determining the reconstruction loss. If None, MSELoss is used.

    """

    LOSS_FUNCTION_ADVERSERIAL = nn.MSELoss(reduction='none')
    LOSS_FUNCTION_ENCODER = nn.MSELoss(reduction='none')
    LOSS_FUNCTION_BCE = nn.BCELoss(reduction='none')

    def __init__(
            self,
            device: str = 'cpu',
            weight_adverserial_loss: float = 1,
            weight_contextual_loss: float = 1,
            weight_encoder_loss: float = 1,
            reconstruction_loss_function: _Loss = None):

        self.device = device
        self.weight_adverserial_loss = weight_adverserial_loss
        self.weight_contextual_loss = weight_contextual_loss
        self.weight_encoder_loss = weight_encoder_loss
        self.loss_function_contextual = reconstruction_loss_function

        self._adverserial_losses = []
        self._contextual_losses = []
        self._encoder_losses = []
        self._generator_losses = []
        self._discriminator_losses = []

    @property
    def adverserial_loss(self):
        """ Gets the collected adverserial loss values over several epochs.

        Returns None if no loss value were collected.
        :rtype : Optional[float]
        """

        if not self._adverserial_losses:
            return None

        return np.array(self._adverserial_losses).mean()

    @property
    def contextual_loss(self):
        """ Gets the collected contextual loss values over several epochs.

        Returns None if no loss value were collected.
        :rtype : Optional[float]
        """

        if not self._contextual_losses:
            return None

        return np.array(self._contextual_losses).mean()

    @property
    def encoder_loss(self):
        """ Gets the collected encoder loss values over several epochs.

        Returns None if no loss value were collected.
        :rtype : Optional[float]
        """

        if not self._encoder_losses:
            return None

        return np.array(self._encoder_losses).mean()

    @property
    def generator_loss(self):
        """ Gets the collected generator loss values over several epochs.

        Returns None if no loss value were collected.
        :rtype : Optional[float]
        """
        if not self._generator_losses:
            return None

        return np.array(self._generator_losses).mean()

    @property
    def discriminator_loss(self):
        """ Gets the collected discriminator loss values over several epochs.

        Returns None if no loss value were collected.

        :rtype : Optional[float]
        """
        if not self._discriminator_losses:
            return None

        return np.array(self._discriminator_losses).mean()

    def update_discriminator_loss(
            self,
            classifier_real: torch.Tensor,
            classifier_fake: torch.Tensor):
        """ Update the discriminator loss values.

        :param classifier_real : torch.Tensor
            The classifier output of the discriminator network for the real data.
        :param classifier_fake : torch.Tensor
            The classifier output of the discriminator network for the fake data.

        :return : torch.Tensor
            The mean loss value for a single epoch.
        """

        labels_real = torch.zeros(size=classifier_real.size(), device=self.device)
        labels_fake = torch.ones(size=classifier_fake.size(), device=self.device)

        loss_real = self.LOSS_FUNCTION_BCE(classifier_real, labels_real).mean(axis=1)
        loss_fake = self.LOSS_FUNCTION_BCE(classifier_fake, labels_fake).mean(axis=1)

        discriminator_loss = (loss_real + loss_fake) / 2
        self._discriminator_losses += discriminator_loss.data.numpy().tolist()

        return discriminator_loss.mean()

    def update_generator_loss(
            self,
            inputs: torch.Tensor,
            generator_output: (torch.Tensor, torch.Tensor, torch.Tensor),
            features_real: torch.Tensor,
            features_fake: torch.Tensor):
        """ Update the generator loss values.

        :param inputs : torch.Tensor
            The original input data.
        :param generator_output : torch.Tensor
            The reconstructed inputs.
        :param features_real : torch.Tensor
            The discriminator features of the original data.
        :param features_fake : torch.Tensor
            The discriminator features of the reconstructed data.
        :return : torch.Tensor
            The mean loss value for a single epoch.
        """

        generated_data, latent_input, latent_output = generator_output

        adverserial_loss = \
            self.LOSS_FUNCTION_ADVERSERIAL(features_real, features_fake).mean(axis=1) * self.weight_adverserial_loss

        contextual_loss = self.loss_function_contextual(generated_data, inputs) \
            if self.loss_function_contextual is not None \
            else L1Loss(reduction='none')(generated_data, inputs)
        contextual_loss = contextual_loss.mean(axis=1) * self.weight_contextual_loss

        encoder_loss = self.LOSS_FUNCTION_ENCODER(latent_input, latent_output).mean(axis=1) * self.weight_encoder_loss

        generator_loss = adverserial_loss + contextual_loss + encoder_loss

        self._adverserial_losses += adverserial_loss.data.numpy().tolist()
        self._contextual_losses += contextual_loss.data.numpy().tolist()
        self._encoder_losses += encoder_loss.data.numpy().tolist()
        self._generator_losses += generator_loss.data.numpy().tolist()

        return generator_loss.mean()

    def reset(self):
        """ Reset the current loss values at the end of an epoch. """

        self._adverserial_losses = []
        self._contextual_losses = []
        self._encoder_losses = []
        self._generator_losses = []
        self._discriminator_losses = []
