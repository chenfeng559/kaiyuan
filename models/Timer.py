import mindspore as ms
from mindspore import nn
from models import TimerBackbone
from mindspore import Tensor
import numpy as np

class Model(nn.Cell):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.ckpt_path = configs.ckpt_path
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout

        self.output_attention = configs.output_attention

        self.backbone = TimerBackbone.Model(configs)
        # Decoder
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.proj
        self.enc_embedding = self.backbone.patch_embedding

<<<<<<< HEAD
        if self.ckpt_path != '':
=======
        if self.ckpt_path:
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
            if self.ckpt_path == 'random':
                print('loading model randomly')
            else:
                print('loading model: ', self.ckpt_path)
<<<<<<< HEAD
                # ckpt_path_str = os.path.abspath(self.ckpt_path)
                # if self.ckpt_path.endswith('.pth'):
                #     #self.backbone.load_state_dict(torch.load(ckpt_path_str, map_location="cpu"))
                # elif self.ckpt_path.endswith('.ckpt'):
                #     # Convert ckpt_path to string for compatibility
                #     # ckpt_path_str = 'checkpoints/Timer_forecast_1.0.ckpt'
                #     # print("Checkpoint path:", ckpt_path_str)
                #     # sd = torch.load(ckpt_path_str, map_location="cpu")["state_dict"]
                #     # sd = {k[6:]: v for k, v in sd.items()}
                #     # self.backbone.load_state_dict(sd, strict=True)
                #     # 构造检查点路径
                #     ckpt_path = os.path.join('checkpoints', 'Timer_forecast_1.0.ckpt')
                #     print("Checkpoint path:", ckpt_path)
                #
                #     # 加载模型状态字典
                #     sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
                #
                #     # 处理可能的前缀
                #     sd = {k.replace('module.', ''): v for k, v in sd.items()}
                #
                #     # 加载状态字典到模型
                #     self.backbone.load_state_dict(sd, strict=False)
                # else:
                #     raise NotImplementedError
        else:
            print('No checkpoint path provided; model not loaded.')

    def encoder_top(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
=======
                # TODO: Add checkpoint loading logic here

    def encoder_top(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdims=True)
        x_enc = x_enc - means
        stdev = ms.ops.sqrt(ms.ops.var(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.transpose(0, 2, 1)  # [B, M, T]
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        dec_in, n_vars = self.enc_embedding(x_enc)

        return dec_in

    def encoder_bottom(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
<<<<<<< HEAD
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        # Encoder
        dec_out, attns = self.decoder(dec_in) # [B * M, N, D]
=======
        means = x_enc.mean(1, keepdims=True)
        x_enc = x_enc - means
        stdev = ms.ops.sqrt(ms.ops.var(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.transpose(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)

        dec_out, attns = self.decoder(dec_in)
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape

<<<<<<< HEAD
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        # Encoder
        dec_out, attns = self.decoder(dec_in) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
=======
        means = x_enc.mean(1, keepdims=True)
        x_enc = x_enc - means
        stdev = ms.ops.sqrt(ms.ops.var(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.transpose(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)

        dec_out, attns = self.decoder(dec_in)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M]

>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape
<<<<<<< HEAD
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        # Encoder
        dec_out, attns = self.decoder(dec_in) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
=======
        means = ms.ops.sum(x_enc, axis=1) / ms.ops.sum(mask == 1, axis=1)
        means = means.unsqueeze(1)
        x_enc = x_enc - means
        x_enc = ms.ops.masked_fill(x_enc, mask == 0, 0)
        stdev = ms.ops.sqrt(ms.ops.sum(x_enc * x_enc, axis=1) / ms.ops.sum(mask == 1, axis=1) + 1e-5)
        stdev = stdev.unsqueeze(1)
        x_enc /= stdev

        x_enc = x_enc.transpose(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)

        dec_out, attns = self.decoder(dec_in)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M]

>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        dec_out = dec_out * stdev + means
        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape

<<<<<<< HEAD
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        # Encoder
        dec_out, attns = self.decoder(dec_in) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
=======
        means = x_enc.mean(1, keepdims=True)
        x_enc = x_enc - means
        stdev = ms.ops.sqrt(ms.ops.var(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.transpose(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)

        dec_out, attns = self.decoder(dec_in)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M]

>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        dec_out = dec_out * stdev + means
        return dec_out

    def predict(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
<<<<<<< HEAD
        # Move model to the correct device
        device = next(self.parameters()).device
        self.to(device)

        # Ensure inputs are on the correct device
        x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(
            device)

        # Normalization from Non-stationary Transformer
        B, L, M = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)  # [B * M, N, D]

        # Encoder
        dec_out, attns = self.decoder(dec_in)  # [B * M, N, D]
        dec_out = self.proj(dec_out)  # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M]

        # De-Normalization from Non-stationary Transformer
=======
        device = next(self.parameters()).device
        self.to(device)

        x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)

        B, L, M = x_enc.shape
        means = x_enc.mean(1, keepdims=True)
        x_enc = x_enc - means
        stdev = ms.ops.sqrt(ms.ops.var(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.transpose(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)

        dec_out, attns = self.decoder(dec_in)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M]

>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        dec_out = dec_out * stdev + means

        return dec_out  # [B, T, D]

<<<<<<< HEAD
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, T, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, T, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, T, D]
        if self.task_name == 'predict':
            dec_out = self.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, T, D]
=======
    def construct(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'predict':
            return self.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        raise NotImplementedError
