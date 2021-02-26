import torch
from torch import nn
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss, L1Loss


class GANomalyLoss:
    LOSS_FUNCTION_ADVERSERIAL = nn.MSELoss(reduction='mean')
    LOSS_FUNCTION_ENCODER = nn.MSELoss(reduction='mean')
    LOSS_FUNCTION_BCE = nn.BCELoss(reduction='mean')

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

        self.adverserial_loss_epoch = 0
        self.contextual_loss_epoch = 0
        self.encoder_loss_epoch = 0
        self.generator_loss_epoch = 0
        self.discriminator_loss_epoch = 0

    def update_discriminator_loss(
            self,
            classifier_real: torch.Tensor,
            classifier_fake: torch.Tensor):
        labels_real = torch.zeros(size=classifier_real.size(), device=self.device)
        labels_fake = torch.ones(size=classifier_fake.size(), device=self.device)

        loss_real = self.LOSS_FUNCTION_BCE(classifier_real, labels_real)
        loss_fake = self.LOSS_FUNCTION_BCE(classifier_fake, labels_fake)

        discriminator_loss = (loss_real + loss_fake) * 0.5
        self.discriminator_loss_epoch = (self.discriminator_loss_epoch + discriminator_loss) * 0.5

        return discriminator_loss

    def update_generator_loss(
            self,
            inputs: torch.Tensor,
            generator_output: (torch.Tensor, torch.Tensor, torch.Tensor),
            features_real: torch.Tensor,
            features_fake: torch.Tensor):
        generated_data, latent_input, latent_output = generator_output

        adverserial_loss = self.LOSS_FUNCTION_ADVERSERIAL(features_real, features_fake)

        contextual_loss = self.loss_function_contextual(generated_data, inputs) \
            if self.loss_function_contextual is not None \
            else L1Loss(reduction='mean')(generated_data, inputs)

        encoder_loss = self.LOSS_FUNCTION_ENCODER(latent_input, latent_output)

        generator_loss = \
            self.weight_adverserial_loss * adverserial_loss \
            + self.weight_contextual_loss * contextual_loss \
            + self.weight_encoder_loss * encoder_loss

        self.adverserial_loss_epoch = (self.adverserial_loss_epoch + adverserial_loss) * 0.5
        self.contextual_loss_epoch = (self.contextual_loss_epoch + contextual_loss) * 0.5
        self.encoder_loss_epoch = (self.encoder_loss_epoch + encoder_loss) * 0.5
        self.generator_loss_epoch = (self.generator_loss_epoch + generator_loss) * 0.5

        return generator_loss

    def reset(self):
        self.adverserial_loss_epoch = 0
        self.contextual_loss_epoch = 0
        self.encoder_loss_epoch = 0
        self.generator_loss_epoch = 0
        self.discriminator_loss_epoch = 0
