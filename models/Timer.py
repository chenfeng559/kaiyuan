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

        if self.ckpt_path:
            if self.ckpt_path == 'random':
                print('loading model randomly')
            else:
                print('loading model: ', self.ckpt_path)
                # TODO: Add checkpoint loading logic here

    def encoder_top(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdims=True)
        x_enc = x_enc - means
        stdev = ms.ops.sqrt(ms.ops.var(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.transpose(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)

        return dec_in

    def encoder_bottom(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdims=True)
        x_enc = x_enc - means
        stdev = ms.ops.sqrt(ms.ops.var(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.transpose(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)

        dec_out, attns = self.decoder(dec_in)
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
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

        dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape
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

        dec_out = dec_out * stdev + means
        return dec_out

    def anomaly_detection(self, x_enc):
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

        dec_out = dec_out * stdev + means
        return dec_out

    def predict(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
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

        dec_out = dec_out * stdev + means

        return dec_out  # [B, T, D]

    def construct(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'predict':
            return self.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise NotImplementedError
