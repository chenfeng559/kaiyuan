import mindspore as ms
from mindspore import nn
from mindspore import Tensor
import mindspore.ops as P
import mindspore.nn.functional as F


class ConvLayer(nn.Cell):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=c_in,
                                   kernel_size=3,
                                   pad_mode='pad',  # Using pad_mode to specify padding mode
                                   padding=2)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

<<<<<<< HEAD
    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
=======
    def construct(self, x):
        x = self.downConv(x.transpose(1, 2))  # [B, C, L]
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)  # [B, L, C]


class EncoderLayer(nn.Cell):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
<<<<<<< HEAD
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
=======
        self.norm1 = nn.LayerNorm([d_model])  # LayerNorm expects shape
        self.norm2 = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def construct(self, x, attn_mask=None, tau=None, delta=None):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

<<<<<<< HEAD
        y = x = self.norm1(x)
=======
        y = self.norm1(x)
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


<<<<<<< HEAD
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
=======
class Encoder(nn.Cell):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.CellList(attn_layers)
        self.conv_layers = nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def construct(self, x, attn_mask=None, tau=None, delta=None):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


<<<<<<< HEAD
class DecoderLayer(nn.Module):
=======
class DecoderLayer(nn.Cell):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
<<<<<<< HEAD
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
=======
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.norm3 = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def construct(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

<<<<<<< HEAD
        y = x = self.norm2(x)
=======
        y = self.norm2(x)
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


<<<<<<< HEAD
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
=======
class Decoder(nn.Cell):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer
        self.projection = projection

    def construct(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
