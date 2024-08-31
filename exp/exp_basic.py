import os
<<<<<<< HEAD

import torch
=======
import mindspore as ms
from mindspore import context
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c

from models import TrmEncoder, Timer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TrmEncoder': TrmEncoder,
            'Timer': Timer,
        }
<<<<<<< HEAD
        if self.args.use_multi_gpu:
            self.model = self._build_model()
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            self.device = self._acquire_device()
            self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
=======

        # Set device context
        if self.args.use_multi_gpu:
            self.device = ms.context.MultiGPU()
            self.model = self._build_model()
        else:
            self.device = self._acquire_device()
            self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError("Model building not implemented")
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
<<<<<<< HEAD
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
=======
            os.environ["DEVICE_ID"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            context.set_context(device_id=int(os.environ["DEVICE_ID"]), mode=context.GRAPH_MODE, device_target="GPU")
            device = "GPU:{}".format(self.args.gpu)
            print('Use GPU: {}'.format(device))
        else:
            context.set_context(device_target="CPU")
            device = "CPU"
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
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
