import math
import mindspore as ms
from mindspore import nn
from mindspore import Tensor

class PositionalEmbedding(nn.Cell):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 计算位置编码
        pe = Tensor(np.zeros((max_len, d_model)), ms.float32)

        position = Tensor(np.arange(0, max_len).astype(np.float32)).reshape(-1, 1)
        div_term = (Tensor(np.arange(0, d_model, 2).astype(np.float32)) *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = ms.numpy.sin(position * div_term)
        pe[:, 1::2] = ms.numpy.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def construct(self, x):
        return self.pe[:, :x.shape[1]]

class TokenEmbedding(nn.Cell):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1  # MindSpore 默认不支持 padding_mode='circular'
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, pad_mode='pad', padding=padding, bias=False)
        self.init_weights()

    def init_weights(self):
        for m in self.get_parameters():
            if isinstance(m, nn.Conv1d):
                nn.initializer.HeNormal()(m.weight)

    def construct(self, x):
        x = self.tokenConv(x.transpose(0, 2, 1))  # [Batch, Time, Feature]
        return x.transpose(0, 2, 1)

class FixedEmbedding(nn.Cell):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = Tensor(np.zeros((c_in, d_model)), ms.float32)

        position = Tensor(np.arange(0, c_in).astype(np.float32)).reshape(-1, 1)
        div_term = (Tensor(np.arange(0, d_model, 2).astype(np.float32)) *
                    -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = ms.numpy.sin(position * div_term)
        w[:, 1::2] = ms.numpy.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model, embedding_init=w)

    def construct(self, x):
        return self.emb(x)

class TemporalEmbedding(nn.Cell):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def construct(self, x):
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class DataEmbedding(nn.Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_inverted(nn.Cell):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Dense(c_in, d_model)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x, x_mark):
        x = x.transpose(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(ms.numpy.concatenate((x, x_mark.transpose(0, 2, 1)), axis=1))
        return self.dropout(x)

class PatchEmbedding(nn.Cell):
    def __init__(self, d_model, patch_len, stride, padding, dropout, position_embedding=True):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.Pad(padding=(0, padding))

        self.value_embedding = nn.Dense(patch_len, d_model, has_bias=False)
        self.positioned = position_embedding

        if position_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x):
        n_vars = x.shape[1]  # [B, M, T]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, M, N, L]
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # reshape

        if self.positioned:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x)
        return self.dropout(x), n_vars
