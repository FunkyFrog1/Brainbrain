import models
import torch
from omegaconf import OmegaConf
import numpy as np
from scipy import signal, stats


def get_stft(x, fs, clip_fs=-1, normalizing=None, **kwargs):
    f, t, Zxx = signal.stft(x, fs, **kwargs)

    Zxx = Zxx[:clip_fs]
    f = f[:clip_fs]

    Zxx = np.abs(Zxx)
    clip = 5  # To handle boundary effects
    if normalizing == "zscore":
        Zxx = Zxx[:, clip:-clip]
        Zxx = stats.zscore(Zxx, axis=-1)
        t = t[clip:-clip]
    # elif normalizing == "baselined":
    #     Zxx = baseline(Zxx[:, clip:-clip])
    #     t = t[clip:-clip]
    elif normalizing == "db":
        Zxx = np.log2(Zxx[:, clip:-clip])
        t = t[clip:-clip]

    if np.isnan(Zxx).any():
        import pdb;
        pdb.set_trace()

    return f, t, Zxx


def build_model(cfg):
    ckpt_path = cfg.upstream_ckpt
    init_state = torch.load(ckpt_path)
    upstream_cfg = init_state["model_cfg"]
    upstream = models.build_model(upstream_cfg)
    return upstream


def load_model_weights(model, states, multi_gpu):
    if multi_gpu:
        model.module.load_weights(states)
    else:
        model.load_weights(states)


def main():
    ckpt_path = "../pretrain_weights/stft_large_pretrained.pth"
    cfg = OmegaConf.create({"upstream_ckpt": ckpt_path})
    model = build_model(cfg)
    model.to('cuda')
    model.eval()
    init_state = torch.load(ckpt_path)
    load_model_weights(model, init_state['model'], False)

    wav = np.load("example_data.npy")[0][0][:2000]

    f, t, linear = get_stft(wav, 2000, clip_fs=40, nperseg=400, noverlap=350, normalizing="zscore",
                            return_onesided=True)  # TODO hardcode sampling rate
    inputs = torch.FloatTensor(linear).unsqueeze(0).transpose(1, 2).to('cuda')
    mask = torch.zeros((inputs.shape[:2])).bool().to('cuda')
    with torch.no_grad():
        output = model.forward(inputs, mask, intermediate_rep=True)

    print("input shape: ", inputs.shape)
    print("output shape: ", output.shape)


if __name__ == '__main__':
    main()