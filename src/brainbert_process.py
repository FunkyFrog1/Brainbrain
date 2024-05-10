from BrainBERT import models
import torch
from omegaconf import OmegaConf
import numpy as np
from scipy import signal, stats


class BrainBERTProcessor(torch.nn.Module):
    def __init__(self, model_path="../pretrain_weights/stft_large_pretrained.pth"):
        super().__init__()
        self.ckpt_path = model_path
        self.cfg = OmegaConf.create({"upstream_ckpt": self.ckpt_path})
        self.model = self.build_model(self.cfg)
        self.model.eval()
        init_state = torch.load(self.ckpt_path)
        self.load_model_weights(self.model, init_state['model'], False)

    def get_stft(self, x, fs, clip_fs=-1, normalizing=None, **kwargs):
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
        return f, t, Zxx

    def build_model(self, cfg):
        ckpt_path = cfg.upstream_ckpt
        init_state = torch.load(ckpt_path)
        upstream_cfg = init_state["model_cfg"]
        upstream = models.build_model(upstream_cfg)
        return upstream

    def load_model_weights(self, model, states, multi_gpu):
        if multi_gpu:
            model.module.load_weights(states)
        else:
            model.load_weights(states)

    def forward(self, wav):
        device = wav.device
        wav = wav.cpu().numpy()
        inputs = []
        for i in range(wav.shape[0]):
            f, t, linear = self.get_stft(wav[i], 2000, clip_fs=40, nperseg=400, noverlap=350, normalizing="zscore",
                                          return_onesided=True)  # TODO hardcode sampling rate
            inputs.append(torch.FloatTensor(linear).transpose(0, 1).to(device))
        inputs = torch.stack(inputs, dim=0).half()
        mask = torch.zeros((inputs.shape[:2])).bool().to(device)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = self.model.forward(inputs, mask, intermediate_rep=True)
        return output

