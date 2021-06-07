import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import warnings

def get_enum(reduction: str) -> int:
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret

def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> str:
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret

def legacy_get_enum(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> int:
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))

def triplet_margin_with_distance_loss(anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    distance_function: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean"
) -> torch.Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )

    tens_ops = (anchor, positive, negative)
    if any([type(t) is not torch.Tensor for t in tens_ops]) and torch.overrides.has_torch_function(tens_ops):
        return torch.overrides.handle_torch_function(
            triplet_margin_with_distance_loss,
            tens_ops,
            anchor,
            positive,
            negative,
            distance_function=distance_function,
            margin=margin,
            swap=swap,
            reduction=reduction,
        )

    distance_function = distance_function if distance_function is not None else nn.PairwiseDistance()

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = torch.min(negative_dist, swap_dist)

    output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)

    reduction_enum = get_enum(reduction)
    if reduction_enum == 1:
        return output.mean()
    elif reduction_enum == 2:
        return output.sum()
    else:
        return output

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class TripletMarginWithDistanceLoss(_Loss):
    def __init__(self, *, distance_function: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 margin: float = 1.0, swap: bool = False, reduction: str = 'mean'):
        super(TripletMarginWithDistanceLoss, self).__init__(size_average=None, reduce=None, reduction=reduction)
        self.distance_function = distance_function if distance_function is not None else nn.PairwiseDistance()
        self.margin = margin
        self.swap = swap

    def forward(self, anchor, positive, negative) -> torch.Tensor:
        return triplet_margin_with_distance_loss(anchor, positive, negative,
                                                   distance_function=self.distance_function,
                                                   margin=self.margin, swap=self.swap, reduction=self.reduction)