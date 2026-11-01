"""Value-dependent checks for eager evaluation, including automatic vmap."""

import torch


class _Check(torch.autograd.Function):
    @staticmethod
    def forward(condition, message):
        if not condition.all():
            raise ValueError(message)
        # A Tensor output lets vmap preserve the condition's batch dimension.
        return condition.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Boolean conditions carry no gradients or saved state.
        pass

    @staticmethod
    def vmap(info, in_dims, condition, message):
        # vmap exposes its batch axis here. Applying the operation again also
        # handles any remaining outer vmap levels before inspecting values.
        return _Check.apply(condition, message), in_dims[0]


def check(condition: torch.Tensor, message: str) -> None:
    """Raise ValueError unless every element of a boolean Tensor is True.

    May be called inside Model.evaluate without consuming extra batch axes.
    If any mapped sample fails, the complete call raises with the given message.
    This eager check reads Tensor values; torch.compile is not guaranteed and
    checking a CUDA condition requires synchronization with the host.
    """
    if not isinstance(condition, torch.Tensor) or condition.dtype != torch.bool:
        raise TypeError("check expects a boolean Tensor")
    if not isinstance(message, str):
        raise TypeError("check message must be a string")
    _Check.apply(condition, message)
