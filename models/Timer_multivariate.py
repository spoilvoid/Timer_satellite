import torch
from torch import nn

from models import TimerBackbone_multivariate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.ckpt_path = configs.ckpt_path
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention

        self.backbone = TimerBackbone_multivariate.Model(configs)
        # Decoder
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.proj
        self.enc_embedding = self.backbone.patch_embedding


        if self.ckpt_path != '':
            if self.ckpt_path == 'random':
                print('loading model randomly')
            else:
                print('loading model: ', self.ckpt_path)
                if self.ckpt_path.endswith('.pth'):
                    model_state_dict = torch.load(self.ckpt_path)
                    # print("here:", list(model_state_dict.keys())[0])
                    if 'module' in list(model_state_dict.keys())[0]:
                        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
                    elif 'model' in list(model_state_dict.keys())[0]:
                        model_state_dict = {k[6:]: v for k, v in model_state_dict.items()}
                    # print(model_state_dict.keys())
                    self.load_state_dict(model_state_dict)
                elif self.ckpt_path.endswith('.ckpt'):
                    sd = torch.load(self.ckpt_path, map_location="cpu")
                    if 'state_dict' in sd.keys():
                        sd = sd['state_dict']
                    sd = {k[6:]: v for k, v in sd.items()}
                    self.backbone.load_state_dict(sd, strict=True)
                else:
                    raise NotImplementedError

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, n_pred_vars=None):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc /= stdev

        batch_size, seq_len, n_vars = x_enc.shape

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [batch_size, n_vars, seq_len]
        dec_in, n_vars, n_tokens = self.enc_embedding(x_enc) # [batch_size, n_vars*n_tokens, d_model]
        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in, n_vars=n_vars, n_tokens=n_tokens, n_pred_vars=n_pred_vars) # [batch_size, n_vars*n_tokens, d_model]
        dec_out = self.proj(dec_out) # [batch_size, n_vars*n_tokens, patch_len]
        dec_out = dec_out.reshape(batch_size, n_vars, -1).transpose(1, 2) # [batch_size, seq_len, n_vars]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, n_pred_vars=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                            torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc /= stdev

        batch_size, seq_len, n_vars = x_enc.shape

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [batch_size, n_vars, seq_len]
        dec_in, n_vars, n_tokens = self.enc_embedding(x_enc) # [batch_size, n_vars*n_tokens, d_model]

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in, n_vars=n_vars, n_tokens=n_tokens, n_pred_vars=n_pred_vars) # [batch_size, n_vars*n_tokens, d_model]
        dec_out = self.proj(dec_out) # [batch_size, n_vars*n_tokens, patch_len]
        dec_out = dec_out.reshape(batch_size, n_vars, -1).transpose(1, 2) # [batch_size, seq_len, n_vars]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * stdev + means
        return dec_out

    def anomaly_detection(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc /= stdev

        batch_size, seq_len, n_vars = x_enc.shape

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # batch_size, n_vars, seq_len]
        dec_in, n_vars, n_tokens = self.enc_embedding(x_enc) # [batch_size, n_vars*n_tokens, d_model]

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in, n_vars=n_vars, n_tokens=n_tokens) # [batch_size, n_vars*n_tokens, d_model]
        dec_out = self.proj(dec_out) # [batch_size, n_vars*n_tokens, patch_len]
        dec_out = dec_out.reshape(batch_size, n_vars, -1).transpose(1, 2) # [batch_size, seq_len, n_vars]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * stdev + means
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, n_pred_vars=None):
        if self.task_name == 'forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask, n_pred_vars=n_pred_vars)
            return dec_out  # [B, T, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, T, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, T, D]

        raise NotImplementedError

