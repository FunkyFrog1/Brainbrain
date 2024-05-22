import random
import math
from torch.nn import MSELoss
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from functools import partial
from loguru import logger
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from dataloader import create_dataloaders, SeegDataset

from seeg_encoder import SeegEncoder
from matplotlib import pyplot as plt

mse_loss = MSELoss()
nll_loss = torch.nn.NLLLoss()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(similarity_matrix, topk=(1, 5)):
    maxk = max(topk)
    batch_size = similarity_matrix.shape[0]

    _, pred = similarity_matrix.topk(maxk, 1, True, True)

    indices = torch.arange(pred.shape[0])

    label = indices.view(-1, 1).repeat(1, pred.shape[1]).to(pred.device)

    correct = pred.eq(label.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:,:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def plot(train_loss_list, val_loss_list, acc_1_list, acc_5_list, avg_weights, similarity_matrix, lr_list=None, params=None, train_name=None, show_lr=True):
    x = np.linspace(1, len(train_loss_list), len(val_loss_list))
    fig = plt.figure(figsize=(16, 9))  # 调整画布大小

    if show_lr:
        lr_x = np.linspace(1, len(lr_list), len(lr_list))

        # 子图1：展示loss
        ax1 = plt.subplot2grid((10, 5), (0, 0), colspan=2, rowspan=3)
        ax1.plot(x, train_loss_list, label='Train Loss')
        ax1.plot(x, val_loss_list, label='Validation Loss')
        ax1.set_title('Loss')
        ax1.legend()

        # 子图2：展示acc
        ax2 = plt.subplot2grid((10, 5), (3, 0), colspan=2, rowspan=3)
        ax2.plot(x, acc_1_list, label='Acc 1')
        ax2.plot(x, acc_5_list, label='Acc 5')
        ax2.set_title('Accuracy')
        ax2.legend()

        # 子图3：展示avg_weights
        ax3 = plt.subplot2grid((10, 5), (9, 0), colspan=5)
        im3 = ax3.imshow(avg_weights, aspect='auto')  # 使用'viridis'颜色映射
        ax3.set_title('Time weight')
        fig.colorbar(im3, ax=ax3)

        # 子图4：展示similarity_matrix
        ax4 = plt.subplot2grid((10, 5), (0, 2), colspan=3, rowspan=9)  # 让子图4占据两行
        im4 = ax4.imshow(similarity_matrix, aspect='equal')  # 使用'plasma'颜色映射
        ax4.set_title('Similarity matrix')
        fig.colorbar(im4, ax=ax4, shrink=0.5)  # 添加shrink参数来调整颜色条的宽度

        ax5 = plt.subplot2grid((10, 5), (6, 0), colspan=2, rowspan=3)
        ax5.plot(lr_x, lr_list, label='learning rate')
        ax5.set_title('Learning rate')
        ax5.legend()

    else:
        # 子图1：展示loss
        ax1 = plt.subplot2grid((9, 5), (0, 0), colspan=2, rowspan=4)
        ax1.plot(x, train_loss_list, label='Train Loss')
        ax1.plot(x, val_loss_list, label='Validation Loss')
        ax1.set_title('Loss')
        ax1.legend()

        # 子图2：展示acc
        ax2 = plt.subplot2grid((9, 5), (4, 0), colspan=2, rowspan=4)
        ax2.plot(x, acc_1_list, label='Acc 1')
        ax2.plot(x, acc_5_list, label='Acc 5')
        ax2.set_title('Accuracy')
        ax2.legend()

        # 子图3：展示avg_weights
        ax3 = plt.subplot2grid((9, 5), (8, 0), colspan=7)
        im3 = ax3.imshow(avg_weights, aspect='auto')  # 使用'viridis'颜色映射
        ax3.set_title('Time weight')
        fig.colorbar(im3, ax=ax3)

        # 子图4：展示similarity_matrix
        ax4 = plt.subplot2grid((9, 5), (0, 2), colspan=3, rowspan=8)  # 让子图4占据两行
        im4 = ax4.imshow(similarity_matrix, aspect='equal')  # 使用'plasma'颜色映射
        ax4.set_title('Similarity matrix')
        fig.colorbar(im4, ax=ax4, shrink=0.5)  # 添加shrink参数来调整颜色条的宽度

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    if params:
        plt.savefig(f'../log/{train_name}/epoch{params["epoch"]}_bs{params["batch"]}_lr{params["lr"]}_tmax{params["t_max"]}_etamin{params["eta_min"]}.png')

    plt.show()


def itm(seeg_embeds, img_embeds, similarity_matrix):
    bs = seeg_embeds.shape[0]
    weight_s2i = F.softmax(similarity_matrix, dim=1)
    weight_i2s = F.softmax(similarity_matrix.t(), dim=1)

    weight_s2i.fill_diagonal_(0)
    weight_i2s.fill_diagonal_(0)

    logit_list = []
    for b in range(bs):
        neg_img_idx = torch.multinomial(weight_s2i[b], 1).item()
        embeds = torch.stack([img_embeds[b], img_embeds[neg_img_idx]])

        logit_list.append(seeg_embeds[b] @ embeds.t())

    for b in range(bs):
        neg_seeg_idx = torch.multinomial(weight_i2s[b], 1).item()
        embeds = torch.stack([seeg_embeds[b], seeg_embeds[neg_seeg_idx]])

        logit_list.append(img_embeds[b] @ embeds.t())

    logits = torch.stack(logit_list)
    labels = torch.Tensor([[1, 0]] * (2 * bs)).to(logits.device)

    # print(nll_loss(logits, labels))
    return F.cross_entropy(logits, labels)



def loss_fn(pred_x, x, temperature=1):
    # loss = mse_loss(pred_x.float(), x.float())
    similarity_matrix = pred_x.float() @ (x.float() / temperature).t()
    targets = torch.arange(x.shape[0]).to(x.device)
    itc_loss = (
        F.cross_entropy(similarity_matrix, targets, label_smoothing=0.1) +
        F.cross_entropy(similarity_matrix.t(), targets, label_smoothing=0.1)
    ) / 2
    # itm_loss = itm(pred_x, x, similarity_matrix)
    loss = itc_loss# + itm_loss
    return loss


def train_fn(seeg_encoder, optimizer, loss_func, train_loader, scheduler):
    scaler = GradScaler()
    seeg_encoder.train()
    train_loss_list = []
    step_loss_list = []
    lr_list_per_epoch = []
    with autocast():
        for step, (seeg, img) in enumerate(train_loader) :
            electrode_shape = seeg.shape[1]
            img = img.to(device).half()
            for i in range(electrode_shape):
                seeg_se = torch.Tensor(seeg[:, i+1, :, :]).to(device)
                seeg_embed = seeg_encoder(seeg_se)
                optimizer.zero_grad()
                loss = loss_func(F.normalize(seeg_embed), F.normalize(img))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_list.append(loss.item())
                break

            # for name, param in seeg_encoder.time_attn.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad.data.norm(2).item())

            if step % 100 == 0:
                logger.info(f'step_{step}'.ljust(10)+str(np.mean(train_loss_list)))
                # logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
                step_loss_list.append(np.mean(train_loss_list))
                train_loss_list = []
                # print(f"\nseeg_single_eletrode\n{seeg_se.shape}\n{seeg_se}")
                # print("\n", seeg_encoder.time_attn.post_avg)
                # print(f"\nseeg_embed\n{seeg_embed.shape}\n{seeg_encoder.time_attn.post_mlp}", )
            scheduler.step()
            lr_list_per_epoch.append(optimizer.param_groups[0]['lr'])

    return np.mean(step_loss_list), lr_list_per_epoch


def val_fn(seeg_encoder, val_loader, loss_func):
    seeg_encoder.eval()
    with torch.no_grad():
        with autocast():
            seeg_embeds = []
            img_embeds = []
            for step, (seeg, img) in enumerate(val_loader) :
                electrode_shape = seeg.shape[1]
                img = img.to(device).half()
                for i in range(electrode_shape):
                    seeg_se = torch.Tensor(seeg[:, i+1, :, :]).to(device)
                    seeg_embed = seeg_encoder(seeg_se)
                    break

                seeg_embeds.append(seeg_embed)
                img_embeds.append(img)

            seeg_embeds = torch.cat(seeg_embeds, dim=0)
            img_embeds = torch.cat(img_embeds, dim=0)

            loss = loss_func(F.normalize(seeg_embeds), F.normalize(img_embeds))

            similarity_matrix = F.normalize(seeg_embeds) @ F.normalize(img_embeds).t()

            acc_1, acc_5 = calculate_accuracy(similarity_matrix)
    return acc_1, acc_5, loss.item(), similarity_matrix


def train(params, train_name):
    dataset = SeegDataset('../data/paired_data/sub_07_data_embed.h5')
    train_loader, val_loader = create_dataloaders(dataset=dataset, train_batch=params['batch'], val_batch=16)

    seeg_encoder = SeegEncoder().to(device)
    optimizer = optim.AdamW(
        params=seeg_encoder.parameters(),
        lr=params['lr'],
        betas=(params['beta1'], params['beta2']),
        eps=params['eps'],
    )

    # optimizer = optim.SGD(seeg_encoder.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["t_max"], eta_min=params["eta_min"])

    loss_func = partial(loss_fn)

    train_loss_list = []
    val_loss_list = []
    acc_1_list = []
    acc_5_list = []
    lr_list = []
    similarity_matrix = None
    for i in range(params['epoch']):
        train_loss, lr_list_per_epoch = train_fn(
            seeg_encoder=seeg_encoder,
            optimizer=optimizer,
            loss_func=loss_func,
            train_loader=train_loader,
            scheduler=scheduler
        )


        acc_1, acc_5, val_loss, val_matrix = val_fn(
            seeg_encoder=seeg_encoder,
            val_loader=val_loader,
            loss_func=loss_func
        )
        logger.info(f"Epoch {i} | loss:{val_loss} | accuracy:top1 {acc_1} top5 {acc_5}")
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        lr_list.extend(lr_list_per_epoch)
        acc_1_list.append(acc_1)
        acc_5_list.append(acc_5)
        similarity_matrix = val_matrix

        # torch.save(seeg_encoder.state_dict(), f'../log/{train_name}/checkpoint{i}.pth')
        # logger.info(f"save checkpoint in ../log/{train_name}/checkpoint{i}.pth")

    plot(train_loss_list,
         val_loss_list,
         acc_1_list,
         acc_5_list,
         seeg_encoder.time_attn.weights.detach().unsqueeze(0).cpu().numpy(),
         similarity_matrix.cpu().numpy(),
         lr_list,
         params,
         train_name
         )


def main():
    params = {
        'epoch': 50,
        'batch': 64,
        "lr": 4e-6,
        "t_max": 3000,
        "eta_min": 0,
        "beta1": 0.9,
        "beta2": 0.98,
        "eps": 1.0e-6,
    }
    for key in params:
        logger.info(key.ljust(20)+str(params[key]))

    train_name = "no_itm_test"  # TODO
    log_name = f'epoch{params["epoch"]}_bs{params["batch"]}_lr{params["lr"]}_tmax{params["t_max"]}_etamin{params["eta_min"]}'
    logger.add(f'../log/{train_name}/{log_name}.log', format="{time:YYYY-MM-DD HH:MM:SS} | {level} | \t{message}")
    logger.info("train name".ljust(20) + train_name)

    train(params, train_name)


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(42)
    main()
