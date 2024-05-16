import random
from torch.nn import MSELoss
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from functools import partial
from loguru import logger
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from brainbert_process import BrainBERTProcessor
from clip_process import ClipProcessor
from dataloader import create_dataloaders, SeegDataset
from seeg_encoder import SeegEncoder

mse_loss = MSELoss()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(similarity_matrix, topk=(5,10)):
    maxk = max(topk)
    batch_size = similarity_matrix.shape[0]

    _, pred = similarity_matrix.topk(maxk, 1, True, True)

    indices = torch.arange(pred.shape[0])

    label = indices.view(-1, 1).repeat(1, pred.shape[1]).to(pred.device)

    correct = pred.eq(label.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def loss_fn(pred_x, x):
    # loss = mse_loss(pred_x.float(), x.float())
    similarity_matrix = pred_x.float() @ x.float().t()
    targets = torch.arange(x.shape[0]).to(x.device)
    loss_itc = (
                   F.cross_entropy(similarity_matrix.t(), targets, label_smoothing=0.1) +
                   F.cross_entropy(similarity_matrix, targets, label_smoothing=0.1)
               ) / 2
    return loss_itc


def train_fn(seeg_encoder, optimizer, loss_func, clip_processor, brainbert_processor, train_loader):
    scaler = GradScaler()
    seeg_encoder.train()
    train_loss_list = []
    with autocast():
        for step, (seeg, img, _, _, _, _) in enumerate(tqdm(train_loader)) :
            batch_shape, electrode_shape = seeg.shape[0], seeg.shape[1]
            seeg, img = seeg.reshape(-1, seeg.shape[-1]).to(device).half(), img.to(device).half()
            img_embed = clip_processor(img)
            seeg_embed = brainbert_processor(seeg)
            seeg_embed = seeg_embed.reshape(batch_shape, electrode_shape, seeg_embed.shape[1], seeg_embed.shape[2])
            # seeg_embed = seeg_encoder(seeg_embed)
            seeg_embed = F.normalize(seeg_encoder(seeg_embed), dim=-1)
            img_embed = F.normalize(img_embed, dim=-1)

            optimizer.zero_grad()
            loss = loss_func(img_embed, seeg_embed)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            train_loss_list.append(loss.item())

            if step % 10 == 0:
                logger.info(f'step_{step}'.ljust(10)+str(np.mean(train_loss_list)))
                train_loss_list = []
                # 假设你的模型实例名为 model
                # for name, param in seeg_encoder.time_attn.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.grad)

    return np.mean(train_loss_list)


def val_fn(seeg_encoder, clip_processor, brainbert_processor, val_loader):
    seeg_encoder.eval()
    with torch.no_grad():
        with autocast():
            seeg_embeds = []
            img_embeds = []
            for seeg, img, _, _, _, _ in tqdm(val_loader):
                batch_shape, electrode_shape = seeg.shape[0], seeg.shape[1]
                seeg, img = seeg.reshape(-1, seeg.shape[-1]).to(device).half(), img.to(device).half()
                img_embed = clip_processor(img)
                seeg_embed = brainbert_processor(seeg)
                seeg_embed = seeg_embed.reshape(batch_shape, electrode_shape, seeg_embed.shape[1], seeg_embed.shape[2])

                # seeg_embed = seeg_encoder(seeg_embed)
                seeg_embed = F.normalize(seeg_encoder(seeg_embed), dim=-1)
                img_embed = F.normalize(img_embed, dim=-1)

                seeg_embeds.append(seeg_embed)
                img_embeds.append(img_embed)

            seeg_embeds = torch.cat(seeg_embeds, dim=0)
            img_embeds = torch.cat(img_embeds, dim=0)

            similarity_matrix = seeg_embeds.half() @ img_embeds.t()
            print(similarity_matrix)

            acc_1, acc_5 = calculate_accuracy(similarity_matrix)
    return acc_1, acc_5


def train(params, train_name):
    dataset = SeegDataset('../data/paired_data/sub_07_data.h5')
    train_loader, val_loader = create_dataloaders(dataset=dataset, train_batch=params['batch'])

    seeg_encoder = SeegEncoder().to(device)
    clip_processor = ClipProcessor().to(device).eval()
    brainbert_processor = BrainBERTProcessor().to(device).eval()

    optimizer = optim.AdamW(
        params=seeg_encoder.parameters(),
        lr=params['lr'],
        betas=(params['beta1'], params['beta2']),
        eps=params['eps'],
    )
    # optimizer = optim.SGD(seeg_encoder.parameters(), lr=params['lr'], momentum=0.98)

    loss_func = partial(loss_fn)

    for i in range(params['epoch']):
        # train_loss = train_fn(
        #     seeg_encoder=seeg_encoder,
        #     optimizer=optimizer,
        #     clip_processor=clip_processor,
        #     brainbert_processor=brainbert_processor,
        #     loss_func=loss_func,
        #     train_loader=train_loader,
        # )
        acc_1, acc_5 = val_fn(
            seeg_encoder=seeg_encoder,
            clip_processor=clip_processor,
            brainbert_processor=brainbert_processor,
            val_loader=val_loader,
        )
        # logger.info(f"Epoch {i} | train_loss:{train_loss.item()} | accuracy:top1 {acc_1} top5 {acc_5}")

        # torch.save(seeg_encoder.state_dict(), f'../log/{train_name}/checkpoint{i}.pth')
        # logger.info(f"save checkpoint in ../log/{train_name}/checkpoint{i}.pth")


def main():
    train_name = "baseline_test" # TODO
    logger.add(f'../log/{train_name}/{train_name}.log', format="{time:YYYY-MM-DD HH:MM:SS} | {level} | \t{message}")
    logger.info("train name".ljust(20) + train_name)

    params = {
        'epoch': 20,
        'batch': 4,
        "lr": 4.0e-3,
        "beta1": 0.9,
        "beta2": 0.98,
        "eps": 1.0e-6,
    }
    for key in params:
        logger.info(key.ljust(20)+str(params[key]))

    train(params, train_name)


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(42)
    main()
