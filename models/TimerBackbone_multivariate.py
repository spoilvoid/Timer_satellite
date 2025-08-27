import torch
from torch import nn

from layers.Embed import PatchEmbedding_multivariate
from layers.SelfAttention_Family import AttentionLayer_multivariate, TimeAttention
from layers.Transformer_EncDec import Encoder_multivariate, EncoderLayer_multivariate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name

        self.padding = 0
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.max_len = configs.max_len

        self.mask_flag = configs.mask_flag
        self.binary_bias = configs.binary_bias
        self.covariate = configs.covariate

        # patching and embedding
        self.patch_embedding = PatchEmbedding_multivariate(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.padding,
            dropout=self.dropout,
            max_len=self.max_len
        )

        self.decoder = Encoder_multivariate(
            [
                 EncoderLayer_multivariate(
                    attention=AttentionLayer_multivariate(
                        attention=TimeAttention(
                            mask_flag=self.mask_flag,
                            binary_bias=self.binary_bias,
                            d_model=configs.d_model,
                            num_heads=configs.n_heads,
                            covariate=self.covariate,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=True
                        ),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.proj = nn.Linear(self.d_model, configs.patch_len, bias=True)