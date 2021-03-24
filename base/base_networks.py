from abc import abstractmethod
from collections import OrderedDict
from typing import Sequence, Optional

import torch
from torch import nn
from torch.nn import Module


class BaseModule(nn.Module):
    """ Base class for NNs, implementing torch.nn.Module. """

    def __init__(self, size_z: int, n_features: int, bias: bool):
        super(BaseModule, self).__init__()

        self.size_z = size_z
        self.n_features = n_features
        self.bias = bias


class NonLinearBaseModule(BaseModule):
    """ Base class for non-linear NNs, implementing BaseModule. """

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int], bias: bool):
        super(NonLinearBaseModule, self).__init__(size_z, n_features, bias)

        self.n_hidden_features = n_hidden_features

    def _get_fully_connected_block(self, in_features: int, out_features: int, idx: int, activation: bool):
        block_dict = OrderedDict([(f'fc{idx}', nn.Linear(in_features, out_features, bias=self.bias))])
        if activation:
            block_dict[f'activation{idx}'] = nn.LeakyReLU()

        return block_dict

    def _get_blocks(
            self,
            n_hidden_features: Sequence[int],
            in_features_first: int,
            out_features_last: int) -> OrderedDict:
        blocks = self._get_fully_connected_block(in_features=in_features_first,
                                                 out_features=n_hidden_features[0],
                                                 idx=1,
                                                 activation=True)

        for i, hidden_features in enumerate(n_hidden_features):
            in_features = hidden_features
            out_features = out_features_last
            activation = False
            idx = i + 2

            if i < len(n_hidden_features) - 1:
                activation = True
                out_features = n_hidden_features[i + 1]

            blocks.update(self._get_fully_connected_block(
                in_features,
                out_features,
                idx,
                activation))

        return blocks

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplemented


class Encoder(NonLinearBaseModule):
    """ Base non-linear encoder network, implementing NonLinearBaseModule. """

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int], bias: bool):
        super(Encoder, self).__init__(size_z, n_features, n_hidden_features, bias)

        blocks = self._get_blocks(n_hidden_features=self.n_hidden_features,
                                  in_features_first=self.n_features,
                                  out_features_last=self.size_z)
        self.model = nn.Sequential(blocks)

    def forward(self, x):
        return self.model(x)

    def __getitem__(self, item):
        return self.model[item]


class Decoder(NonLinearBaseModule):
    """ Base non-linear decoder network, implementing NonLinearBaseModule. """

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int], bias: bool):
        super(Decoder, self).__init__(size_z, n_features, n_hidden_features, bias)

        self.n_hidden_features_reversed = self.n_hidden_features[::-1]

        blocks = self._get_blocks(n_hidden_features=self.n_hidden_features_reversed,
                                  in_features_first=self.size_z,
                                  out_features_last=self.n_features)
        self.model = nn.Sequential(blocks)

    def forward(self, x):
        return self.model(x)

    def __getitem__(self, item):
        return self.model[item]

    def add_module(self, name: str, module: Optional['Module']) -> None:
        self.model.add_module(name, module)


class BaseSubNetwork(BaseModule):
    """ Base network for GANomaly and VAE sub-networks, implementing BaseModule. """

    def __init__(
            self,
            size_z: int,
            n_features: int,
            linear: bool,
            n_hidden_features: Sequence[int],
            bias: bool):
        super(BaseSubNetwork, self).__init__(size_z, n_features, bias)

        self.linear = linear
        self.n_hidden_features = n_hidden_features

        if not linear:
            assert all(self.size_z < hidden_features < self.n_features
                       for hidden_features in self.n_hidden_features), \
                'Invalid number of hidden features'
        else:
            assert len(self.n_hidden_features) == 1, 'Invalid number of hidden features'

    def _get_encoder_network(self):
        linear_encoder = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.n_features, self.n_hidden_features[0], bias=self.bias)),
            ('fc2', nn.Linear(self.n_hidden_features[0], self.size_z, bias=self.bias))
        ]))

        return linear_encoder \
            if self.linear \
            else Encoder(self.size_z, self.n_features, self.n_hidden_features, self.bias)

    def _get_decoder_network(self):
        linear_decoder = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.size_z, self.n_hidden_features[0], bias=self.bias)),
            ('fc2', nn.Linear(self.n_hidden_features[0], self.n_features, bias=self.bias))
        ]))

        return linear_decoder \
            if self.linear \
            else Decoder(self.size_z, self.n_features, self.n_hidden_features, self.bias)


class DiscriminatorNet(BaseSubNetwork):
    """ GANomaly discriminator net, implementing BaseSubNetwork.

    Parameters
    ----------
    size_z : int
        The latent space dimensionality.
    n_features : int
        The number of input layer units.
    n_hidden_features : int, default=None
        Is Ignored if liner is True.
        The number of units used in the hidden layers.
    linear : bool, default=True
        Specifies if only linear layers without activation are used.
    bias : bool, default=True
        Specifies if bias terms are used within the layers.

    Attributes
    ----------
    features : torch.nn.Module
        The encoder part of the discriminator.
    classifier : torch.nn.Sequential
        The classification part of the discriminator.
    """

    def __init__(
            self,
            size_z: int,
            n_features: int,
            n_hidden_features: Sequence[int] = None,
            linear: bool = True,
            bias: bool = True):
        super(DiscriminatorNet, self).__init__(size_z, n_features, linear, n_hidden_features, bias)

        self.features = self._get_encoder_network()[:-1]
        self.classifier = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.n_hidden_features[-1], 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x: torch.Tensor):
        """ Return the feedforward result based on the input.

        :param x : torch.Tensor
            The input tensor.

        :return : (torch.Tensor, torch.Tensor)
            [0]: The classification result (real, fake).
            [1]: The features of a subsequent layer.
        """
        features = self.features(x)
        classifier = self.classifier(features)

        return classifier, features


class GeneratorNet(BaseSubNetwork):
    """ Generator net of the GANomaly architecture, comprising an encoder, a decoder and an 2nd encoder.

    Parameters
    ----------
    size_z : int
        The latent space dimensionality.
    n_features : int
        The number of input layer units.
    n_hidden_features : int, default=None
        Is Ignored if liner is True.
        The number of units used in the hidden layers.
    linear : bool, default=True
        Specifies if only linear layers without activation are used.
    bias : bool, default=True
        Specifies if bias terms are used within the layers.

    Attributes
    ----------
    encoder1 : torch.nn.Module
        The 1st encoder.
    decoder : torch.nn.Module
        The decoder.
    encoder2 : torch.nn.Module
        The 2nd encoder.
    """

    def __init__(
            self,
            size_z: int,
            n_features: int,
            n_hidden_features: Sequence[int] = None,
            linear: bool = True,
            bias: bool = True):
        super(GeneratorNet, self).__init__(size_z, n_features, linear, n_hidden_features, bias)

        self.encoder1 = self._get_encoder_network()
        self.decoder = self._get_decoder_network()
        self.encoder2 = self._get_encoder_network()

    def forward(self, x: torch.Tensor):
        """ Return the feedforward result based on the input.

        :param x : torch.Tensor
            The input tensor.

        :return : (torch.Tensor, torch.Tensor, torch.Tensor)
            [0]: The result of the 1st encoder (latent representation of the input z).
            [1]: The reconstructed input (x hat).
            [2]: The result of the 2nd encoder (latent representation of the reconstructed input z hat).
        """
        latent_input = self.encoder1(x)
        generated_data = self.decoder(latent_input)
        latent_output = self.encoder2(generated_data)

        # output autoencoder, output first encoder, output second encoder
        return generated_data, latent_input, latent_output


class MultivariateGaussianEncoder(BaseSubNetwork):
    """ Encoder modeling a multivariate Gaussian.

    Parameters
    ----------
    size_z : int
        The latent space dimensionality.
    n_features : int
        The number of input layer units.
    n_hidden_features : int, default=None
        Is Ignored if liner is True.
        The number of units used in the hidden layers.
    linear : bool, default=True
        Specifies if only linear layers without activation are used.
    bias : bool, default=True
        Specifies if bias terms are used within the layers.

    Attributes
    ----------
    encoder : torch.nn.Module
        The encoder network.
    fc_mean : torch.nn.Linear
        The layer determining the mean of the modeled distribution based on the encoded input.
    fc_variance : torch.nn.Linear
        The layer determining the log_variance of the modeled distribution based on the encoded input.
    """

    def __init__(
            self,
            size_z: int,
            n_features: int,
            n_hidden_features: Sequence[int] = None,
            linear: bool = True,
            bias: bool = True):
        super(MultivariateGaussianEncoder, self).__init__(size_z, n_features, linear, n_hidden_features, bias)

        self.encoder = self._get_encoder_network()[:-1]
        self.fc_mean = nn.Linear(in_features=self.n_hidden_features[-1], out_features=self.size_z, bias=self.bias)
        self.fc_variance = \
            nn.Linear(in_features=self.n_hidden_features[-1], out_features=self.size_z, bias=self.bias)

    def forward(self, x: torch.Tensor):
        """ Return the feedforward result based on the input.

        :param x : torch.Tensor
            The input tensor.

        :return : (torch.Tensor, torch.Tensor)
            [0]: The mean of the multivariate Gaussian.
            [1]: The log variance of the multivariate Gaussian.
        """

        encoder_output = self.encoder(x)
        mean = self.fc_mean(encoder_output)
        log_variance = self.fc_variance(encoder_output)

        return mean, log_variance
