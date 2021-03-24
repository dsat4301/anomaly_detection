import torch
from torch import Tensor
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss


class BaseLoss(_Loss):
    """ Base class for custom loss functions, implementing torch.nn.modules.loss._Loss. """

    def __init__(self, reduction: str = 'mean'):
        super(BaseLoss, self).__init__(None, None, reduction)

    def reduce(self, result: torch.Tensor):
        if self.reduction == 'none':
            return result

        if self.reduction == 'mean':
            return result.mean()

        if self.reduction == 'sum':
            return result.sum()


class ChiSquareLoss(BaseLoss):
    """ Chi-squared-distance-based loss function, applied for distance calculation among histograms.
        Referring to to. Pele, O. & Werman, M. The quadratic-chi histogram distance family
        in European conference on computer vision(Springer, 2010), 749–762.
    """

    def __init__(self, reduction: str = 'mean', alpha: float = 1e-3) -> None:
        super(ChiSquareLoss, self).__init__(reduction)

        self.alpha = alpha

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if not (targets.size() == inputs.size()):
            raise ValueError('Arguments must be of same shape.')

        sums = inputs + targets
        squared_differences = (inputs - targets).pow(2)

        result = 0.5 * squared_differences / (sums + self.alpha)

        assert (result >= 0).all()

        return self.reduce(result)
