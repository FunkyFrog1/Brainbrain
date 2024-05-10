import torch
import torch.nn as nn
from collections import OrderedDict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TimeAttentionBlock(nn.Module):
    def __init__(self, width=768, layers=6, heads=12, time_frame=31, pos_embed=True):
        super().__init__()
        self.pos_embed = pos_embed
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((time_frame+1, width)))
        self.ln = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.proj = nn.Parameter(scale * torch.randn(width, width))

    def forward(self, x):
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        if self.pos_embed:
            x = x + self.positional_embedding.to(x.dtype)
        x = self.ln(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 0, :]
        x = x @ self.proj

        return x


class ElectrodeAttentionBlock(nn.Module):
    def __init__(self, width=768, layers=6, heads=12, electrode_num=147, pos_embed=True):
        super().__init__()
        self.pos_embed = pos_embed
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((electrode_num+1, width)))
        self.ln = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.proj = nn.Parameter(scale * torch.randn(width, width))

    def forward(self, x):
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        if self.pos_embed:
            x = x + self.positional_embedding.to(x.dtype)
        x = self.ln(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 0, :]
        x = x @ self.proj

        return x


class SeegEncoder(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.time_attn = TimeAttentionBlock(width=width)
        self.electrode_attn = ElectrodeAttentionBlock(width=width)
        # self.ln = LayerNorm(width)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_ = x[:, i, :, :]
            x_ = self.time_attn(x_)
            x_list.append(x_)

        x = torch.stack(x_list, dim=1)
        x = self.electrode_attn(x)

        return x


def main():
    model = SeegEncoder().cuda()
    x = torch.randn((4, 147, 31, 768)).cuda()
    output = model(x)


if __name__ == '__main__':
    main()
