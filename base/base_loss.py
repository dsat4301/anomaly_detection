from torch import Tensor
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss


class ChiSquareLoss(_Loss):
    def __init__(self, reduction: str = 'mean', alpha: float = 1e-3) -> None:
        super(ChiSquareLoss, self).__init__(None, None, reduction)

        self.alpha = alpha

    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if not (target.size() == input.size()):
            raise ValueError('Arguments must be of same shape.')

        sums = input + target
        squared_differences = (input - target).pow(2)

        result = 0.5 * squared_differences / (sums + self.alpha)

        assert (result >= 0).all()

        if self.reduction == 'none':
            return result

        if self.reduction == 'mean':
            return result.mean()

        if self.reduction == 'sum':
            return result.sum()
