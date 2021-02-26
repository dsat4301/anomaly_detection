import torch
from torch import Tensor
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss


class BaseLoss(_Loss):
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
    def __init__(self, reduction: str = 'mean', alpha: float = 1e-3) -> None:
        super(ChiSquareLoss, self).__init__(reduction)

        self.alpha = alpha

    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if not (target.size() == input.size()):
            raise ValueError('Arguments must be of same shape.')

        sums = input + target
        squared_differences = (input - target).pow(2)

        result = 0.5 * squared_differences / (sums + self.alpha)

        assert (result >= 0).all()

        return self.reduce(result)


class ChiAbsoluteLoss(BaseLoss):

    def __init__(self, reduction: str = 'mean', alpha: float = 1e-3) -> None:
        super(ChiAbsoluteLoss, self).__init__(reduction)

        self.alpha = alpha

    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if not (target.size() == input.size()):
            raise ValueError('Arguments must be of same shape.')

        sums = input + target
        absolute_differences = torch.abs(input - target)

        result = 0.5 * absolute_differences / (sums + self.alpha)

        assert (result >= 0).all()

        return self.reduce(result)
