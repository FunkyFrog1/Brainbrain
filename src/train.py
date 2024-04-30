import torch
import time
from torch.nn.functional import multi_head_attention_forward
from xformers.ops.fmha import memory_efficient_attention


def test(device):
    # 初始化输入
    q = torch.rand(10000, 128, 8, 16).to(device)
    k = torch.rand(10000, 128, 8, 16).to(device)
    v = torch.rand(10000, 128, 8, 16).to(device)

    # xFormers的memory_efficient_attention
    start_time = time.time()
    xformers_output = memory_efficient_attention(q, k, v)
    print(xformers_output.shape)
    xformers_time = time.time() - start_time

    # PyTorch原生的attention
    start_time = time.time()
    attn_output, _ = multi_head_attention_forward(
        query=q.view(q.shape[0], q.shape[1], -1),
        key=k.view(k.shape[0], k.shape[1], -1),
        value=v.view(v.shape[0], k.shape[1], -1),
        embed_dim_to_check=16*8,
        num_heads=8,
        in_proj_weight=torch.rand(3*16*8, 16*8).to(device),
        in_proj_bias=torch.rand(3*16*8).to(device),
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=torch.rand(128, 128).to(device),
        out_proj_bias=torch.rand(128).to(device),
        training=True,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=True,
        q_proj_weight=torch.rand(128, 128).to(device),
        k_proj_weight=torch.rand(128, 128).to(device),
        v_proj_weight=torch.rand(128, 128).to(device),
        static_k=None,
        static_v=None
    )
    print(attn_output.shape)
    pytorch_time = time.time() - start_time

    print(f"PyTorch attention time: {pytorch_time}")
    print(f"xFormers attention time: {xformers_time}")


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    test(device)