import mindspore as ms
from mindspore import nn
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Cell):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        padding = 0

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, self.dropout)

        # Decoder
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.proj = nn.Dense(self.d_model, configs.patch_len, has_bias=True)

    def construct(self, x):
        # Patching and embedding
        x = self.patch_embedding(x)

        # Decoder forward pass
        x = self.decoder(x)

        # Prediction
        output = self.proj(x)

        return output
