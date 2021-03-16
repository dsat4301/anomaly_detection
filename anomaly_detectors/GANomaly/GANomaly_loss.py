from collections import OrderedDict

import numpy as np
import torch
from torch import nn
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss, L1Loss


class GANomalyLoss:
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

        self.adverserial_loss_epoch = []
        self.contextual_loss_epoch = []
        self.encoder_loss_epoch = []
        self.generator_loss_epoch = []
        self.discriminator_loss_epoch = []

    def update_discriminator_loss(
            self,
            classifier_real: torch.Tensor,
            classifier_fake: torch.Tensor):
        labels_real = torch.zeros(size=classifier_real.size(), device=self.device)
        labels_fake = torch.ones(size=classifier_fake.size(), device=self.device)

        loss_real = self.LOSS_FUNCTION_BCE(classifier_real, labels_real).mean(axis=1)
        loss_fake = self.LOSS_FUNCTION_BCE(classifier_fake, labels_fake).mean(axis=1)

        discriminator_loss = (loss_real + loss_fake) / 2
        self.discriminator_loss_epoch += discriminator_loss.data.numpy().tolist()

        return discriminator_loss.mean()

    def update_generator_loss(
            self,
            inputs: torch.Tensor,
            generator_output: (torch.Tensor, torch.Tensor, torch.Tensor),
            features_real: torch.Tensor,
            features_fake: torch.Tensor):
        generated_data, latent_input, latent_output = generator_output

        adverserial_loss = self.LOSS_FUNCTION_ADVERSERIAL(features_real, features_fake).mean(axis=1)

        contextual_loss = self.loss_function_contextual(generated_data, inputs) \
            if self.loss_function_contextual is not None \
            else L1Loss(reduction='none')(generated_data, inputs)
        contextual_loss = contextual_loss.mean(axis=1)

        encoder_loss = self.LOSS_FUNCTION_ENCODER(latent_input, latent_output).mean(axis=1)

        generator_loss = \
            self.weight_adverserial_loss * adverserial_loss \
            + self.weight_contextual_loss * contextual_loss \
            + self.weight_encoder_loss * encoder_loss

        self.adverserial_loss_epoch += adverserial_loss.data.numpy().tolist()
        self.contextual_loss_epoch += contextual_loss.data.numpy().tolist()
        self.encoder_loss_epoch += encoder_loss.data.numpy().tolist()
        self.generator_loss_epoch += generator_loss.data.numpy().tolist()

        return generator_loss.mean()

    def reset(self):
        self.adverserial_loss_epoch = []
        self.contextual_loss_epoch = []
        self.encoder_loss_epoch = []
        self.generator_loss_epoch = []
        self.discriminator_loss_epoch = []

    def get_mean_epoch_results(self):
        if not self.generator_loss_epoch or not self.discriminator_loss_epoch:
            return None

        return OrderedDict([
            ('adverserial_loss', np.array(self.adverserial_loss_epoch).mean()),
            ('contextual_loss', np.array(self.contextual_loss_epoch).mean()),
            ('encoder_loss', np.array(self.encoder_loss_epoch).mean()),
            ('generator_loss', np.array(self.generator_loss_epoch).mean()),
            ('discriminator_loss', np.array(self.discriminator_loss_epoch).mean())])
