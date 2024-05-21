import h5py
import torch

from dataloader import SeegDataset, DataLoader
from brainbert_process import BrainBERTProcessor
from clip_process import ClipProcessor
from tqdm import tqdm

# 加载数据
dataset = SeegDataset('../data/paired_data/sub_07_data.h5')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 加载模型
brainprocess = BrainBERTProcessor().cuda()
brainprocess.eval()
clipprocess = ClipProcessor().cuda()
clipprocess.eval()

# 创建一个新的h5文件来保存处理后的数据
with h5py.File('../data/paired_data/sub_07_data_embed.h5', 'w') as f:
    data_seeg = []
    data_img = []
    for index, (seeg, img, _, _, _, _) in enumerate(tqdm(dataloader)):
        seeg, img = seeg.cuda(), img.cuda().float()
        seeg_embeds = []
        img_embeds = []
        with torch.no_grad():
            for i in range(seeg.shape[1]):
                seeg_embed = brainprocess(seeg[:,i,:])
                seeg_embeds.append(seeg_embed)

            img = clipprocess(img).cpu()
            embeds = torch.stack(seeg_embeds, dim=1).cpu()

        data_seeg.append(embeds)
        data_img.append(img)

    data_seeg = torch.cat(data_seeg)
    data_img = torch.cat(data_img)
    print(data_seeg.shape, data_seeg.dtype)
    print(data_img.shape, data_img.dtype)

    # 存储 embeds 数据
    f.create_dataset('seeg', data=data_seeg)
    # 存储 img 数据
    f.create_dataset('img', data=data_img)

