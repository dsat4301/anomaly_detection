from collections import OrderedDict
from typing import Sequence

from torch import nn


class BaseModule(nn.Module):

    def __init__(self, size_z: int, n_features: int, bias: bool = True):
        super(BaseModule, self).__init__()

        self.size_z = size_z
        self.n_features = n_features
        self.bias = bias


class EncoderLinear(BaseModule):

    def __init__(self, size_z: int, n_features: int):
        super(EncoderLinear, self).__init__(size_z, n_features)

        self.linear = nn.Linear(self.n_features, self.size_z, bias=self.bias)

    def forward(self, x):
        return self.linear(x)


class DecoderLinear(BaseModule):

    def __init__(self, size_z: int, n_features: int):
        super(DecoderLinear, self).__init__(size_z, n_features)

        self.linear = nn.Linear(self.size_z, self.n_features, bias=self.bias)

    def forward(self, x):
        return self.linear(x)


class NonLinearBaseModule(BaseModule):

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int]):
        super(NonLinearBaseModule, self).__init__(size_z, n_features)

        self.n_hidden_features = n_hidden_features

    def get_fully_connected_block(self, in_features: int, out_features: int, idx: int, activation: bool):
        block_dict = OrderedDict([(f'fc{idx}', nn.Linear(in_features, out_features, bias=self.bias))])
        if activation:
            block_dict[f'activation{idx}'] = nn.LeakyReLU()

        return block_dict

    def get_blocks(
            self,
            n_hidden_features: Sequence[int],
            in_features_first: int,
            out_features_last: int) -> OrderedDict:
        blocks = self.get_fully_connected_block(in_features=in_features_first,
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

            blocks.update(self.get_fully_connected_block(
                in_features,
                out_features,
                idx,
                activation))

        return blocks


class Encoder(NonLinearBaseModule):

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int]):
        super(Encoder, self).__init__(size_z, n_features, n_hidden_features)

        blocks = self.get_blocks(n_hidden_features=self.n_hidden_features,
                                 in_features_first=self.n_features,
                                 out_features_last=self.size_z)
        self.model = nn.Sequential(blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(NonLinearBaseModule):

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int]):
        super(Decoder, self).__init__(size_z, n_features, n_hidden_features)

        self.n_hidden_features_reversed = self.n_hidden_features[::-1]

        blocks = self.get_blocks(n_hidden_features=self.n_hidden_features_reversed,
                                 in_features_first=self.size_z,
                                 out_features_last=self.n_features)
        self.model = nn.Sequential(blocks)

    def forward(self, x):
        return self.model(x)


class BaseSubNetwork(BaseModule):
    def __init__(
            self,
            size_z: int,
            n_features: int,
            linear: bool = True,
            n_hidden_features: Sequence[int] = None):
        super(BaseSubNetwork, self).__init__(size_z, n_features)

        self.linear = linear
        self.n_hidden_features = n_hidden_features

        if not linear:
            assert all(self.size_z < hidden_features < self.n_features
                       for hidden_features in self.n_hidden_features), \
                'Invalid number of hidden features'

    def get_encoder_network(self):
        return EncoderLinear(self.size_z, self.n_features) if self.linear \
            else Encoder(self.size_z, self.n_features, self.n_hidden_features)

    def get_decoder_network(self):
        return DecoderLinear(self.size_z, self.n_features) if self.linear \
            else Decoder(self.size_z, self.n_features, self.n_hidden_features)


class DiscriminatorNet(BaseSubNetwork):
    def __init__(
            self,
            size_z: int,
            n_features: int,
            linear: bool = True,
            n_hidden_features: Sequence[int] = None):
        super(DiscriminatorNet, self).__init__(size_z, n_features, linear, n_hidden_features)

        self.features = self.get_encoder_network()
        self.classifier = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.size_z, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)

        # prediction (anomaly or normal), encoder output
        return classifier, features


class GeneratorNet(BaseSubNetwork):
    def __init__(
            self,
            size_z: int,
            n_features: int,
            linear: bool,
            n_hidden_features: Sequence[int] = None):
        super(GeneratorNet, self).__init__(size_z, n_features, linear, n_hidden_features)

        self.encoder1 = self.get_encoder_network()
        self.decoder = self.get_decoder_network()
        self.encoder2 = self.get_encoder_network()

    def forward(self, x):
        latent_input = self.encoder1(x)
        generated_data = self.decoder(latent_input)
        latent_output = self.encoder2(x)

        # output autoencoder, output first encoder, output second encoder
        return generated_data, latent_input, latent_output
