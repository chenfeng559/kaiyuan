import os
import mindspore as ms
from mindspore import context

from models import TrmEncoder, Timer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TrmEncoder': TrmEncoder,
            'Timer': Timer,
        }

        # Set device context
        if self.args.use_multi_gpu:
            self.device = ms.context.MultiGPU()
            self.model = self._build_model()
        else:
            self.device = self._acquire_device()
            self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError("Model building not implemented")
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["DEVICE_ID"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            context.set_context(device_id=int(os.environ["DEVICE_ID"]), mode=context.GRAPH_MODE, device_target="GPU")
            device = "GPU:{}".format(self.args.gpu)
            print('Use GPU: {}'.format(device))
        else:
            context.set_context(device_target="CPU")
            device = "CPU"
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self, setting):
        pass

    def finetune(self, setting):
        pass

    def test(self, setting, test=0):
        pass
