import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


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
        # self.pos_embed = pos_embed
        scale = width ** -0.5
        self.weights = nn.Parameter(torch.ones(31))
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.BatchNorm1d(768),
        )

        self.post_avg = None
        self.post_mlp = None


    def forward(self, x):
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        # x = self.mlp(x).reshape(x.shape[0], 31, 768)
        # if self.pos_embed:
        #     x = x + self.positional_embedding.to(x.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = x.unsqueeze(2)
        # x = self.mlp(x)
        # x = x.squeeze(1)
        # x = x.squeeze(1)
        weights = F.softmax(self.weights, dim=0)
        x = torch.matmul(weights.unsqueeze(0), x).squeeze()
        self.post_avg = x
        x = self.mlp(x)
        self.post_mlp = x

        return x


class ElectrodeAttentionBlock(nn.Module):
    def __init__(self, width=768, layers=6, heads=12, electrode_num=147, pos_embed=False):
        super().__init__()
        self.pos_embed = pos_embed
        scale = width ** -0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((electrode_num, width)))
        # self.mlp = nn.Conv2d(147, 1, 1)
        self.weights = nn.Parameter(torch.ones(147))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, width))

    def forward(self, x):
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        # if self.pos_embed:
        #     x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = x.unsqueeze(2)
        # x = self.mlp(x)
        # x = x.squeeze(1)
        # x = x.squeeze(1)
        weights = F.softmax(self.weights, dim=0)
        x = torch.matmul(weights.unsqueeze(0), x).squeeze()
        x = self.ln_post(x)
        x = x @ self.proj

        return x

class SeegEncoder(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.time_attn = TimeAttentionBlock(width=width)
        # self.electrode_attn = ElectrodeAttentionBlock(width=width)

    def forward(self, x):
        x = self.time_attn(x)

        return x
def main():
    model = SeegEncoder().cuda()
    x = torch.randn((8, 147, 31, 768)).cuda()
    output = model(x)


if __name__ == '__main__':
    main()
