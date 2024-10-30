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
