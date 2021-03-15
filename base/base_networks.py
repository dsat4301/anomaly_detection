from abc import abstractmethod
from collections import OrderedDict
from typing import Sequence, Optional

from torch import nn
from torch.nn import Module


class BaseModule(nn.Module):

    def __init__(self, size_z: int, n_features: int, bias: bool):
        super(BaseModule, self).__init__()

        self.size_z = size_z
        self.n_features = n_features
        self.bias = bias


class NonLinearBaseModule(BaseModule):

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int], bias: bool):
        super(NonLinearBaseModule, self).__init__(size_z, n_features, bias)

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

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplemented


class Encoder(NonLinearBaseModule):

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int], bias: bool):
        super(Encoder, self).__init__(size_z, n_features, n_hidden_features, bias)

        blocks = self.get_blocks(n_hidden_features=self.n_hidden_features,
                                 in_features_first=self.n_features,
                                 out_features_last=self.size_z)
        self.model = nn.Sequential(blocks)

    def forward(self, x):
        return self.model(x)

    def __getitem__(self, item):
        return self.model[item]


class Decoder(NonLinearBaseModule):

    def __init__(self, size_z: int, n_features: int, n_hidden_features: Sequence[int], bias: bool):
        super(Decoder, self).__init__(size_z, n_features, n_hidden_features, bias)

        self.n_hidden_features_reversed = self.n_hidden_features[::-1]

        blocks = self.get_blocks(n_hidden_features=self.n_hidden_features_reversed,
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

    def get_encoder_network(self):
        linear_encoder = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.n_features, self.n_hidden_features[0], bias=self.bias)),
            ('fc2', nn.Linear(self.n_hidden_features[0], self.size_z, bias=self.bias))
        ]))

        return linear_encoder \
            if self.linear \
            else Encoder(self.size_z, self.n_features, self.n_hidden_features, self.bias)

    def get_decoder_network(self):
        linear_decoder = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.size_z, self.n_hidden_features[0], bias=self.bias)),
            ('fc2', nn.Linear(self.n_hidden_features[0], self.n_features, bias=self.bias))
        ]))

        return linear_decoder \
            if self.linear \
            else Decoder(self.size_z, self.n_features, self.n_hidden_features, self.bias)


class DiscriminatorNet(BaseSubNetwork):

    def __init__(
            self,
            size_z: int,
            n_features: int,
            n_hidden_features: Sequence[int],
            linear: bool = True,
            bias: bool = True):
        super(DiscriminatorNet, self).__init__(size_z, n_features, linear, n_hidden_features, bias)

        self.features = self.get_encoder_network()[:-1]
        self.classifier = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.n_hidden_features[-1], 1)),
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
            n_hidden_features: Sequence[int],
            linear: bool = True,
            bias: bool = True):
        super(GeneratorNet, self).__init__(size_z, n_features, linear, n_hidden_features, bias)

        self.encoder1 = self.get_encoder_network()
        self.decoder = self.get_decoder_network()
        self.encoder2 = self.get_encoder_network()

    def forward(self, x):
        latent_input = self.encoder1(x)
        generated_data = self.decoder(latent_input)
        latent_output = self.encoder2(generated_data)

        # output autoencoder, output first encoder, output second encoder
        return generated_data, latent_input, latent_output


class MultivariateGaussianEncoder(BaseSubNetwork):
    def __init__(
            self,
            size_z: int,
            n_features: int,
            n_hidden_features: Sequence[int],
            linear: bool = True,
            bias: bool = True):
        super(MultivariateGaussianEncoder, self).__init__(size_z, n_features, linear, n_hidden_features, bias)

        self.encoder = self.get_encoder_network()[:-1]
        self.fc_mean = nn.Linear(in_features=self.n_hidden_features[-1], out_features=self.size_z, bias=self.bias)
        self.fc_variance = \
            nn.Linear(in_features=self.n_hidden_features[-1], out_features=self.size_z, bias=self.bias)

    def forward(self, x):
        encoder_output = self.encoder(x)
        mean = self.fc_mean(encoder_output)
        log_variance = self.fc_variance(encoder_output)

        return mean, log_variance
