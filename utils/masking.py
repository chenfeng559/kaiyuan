<<<<<<< HEAD
import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
=======
import mindspore as ms
import mindspore.ops as ops


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        self._mask = ops.Triu()(ops.ones(mask_shape, dtype=ms.bool_), diagonal=1)

        if device == "gpu":
            self._mask = self._mask.to(ms.context.PYNATIVE)

    @property
    def mask(self):
        return self._mask
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
