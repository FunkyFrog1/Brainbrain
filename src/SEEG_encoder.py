import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt


def seeg_plot(seeg):
    """
    此函数用于绘制 SEEG 数据

    输入 seeg (batch, channel, seq_len)
    绘制第一个 batch 上的前18个通道的SEEG信号用于调试

    不返回结果
    """
    seq_len = seeg.shape[-1]
    x = np.linspace(0, seq_len - 1, seq_len)

    for index, channel in enumerate(seeg[0]):
        plt.subplot(6, 3, 0 + index + 1)
        plt.plot(x, channel)
        if index == 17:
            break

    plt.show()


def test(device):
    seeg_raw_data = torch.rand(1, 92, 250)

    seeg_plot(seeg_raw_data)



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    test(device)