import random
from torch.nn import MSELoss
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from functools import partial

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


def loss_fn(pred_x, x):
    loss = mse_loss(pred_x.float(), x.float())
    return loss


def train_fn(seeg_encoder, optimizer, loss_func, clip_processor, brainbert_processor, train_loader):
    seeg_encoder.train()
    train_loss_list = []
    for step, (seeg, img, _, _, _, _) in enumerate(tqdm(train_loader)) :
        batch_shape, electrode_shape = seeg.shape[0], seeg.shape[1]
        seeg, img = seeg.reshape(-1, seeg.shape[-1]).to(device), img.to(device).half()
        img_embed = clip_processor(img)
        seeg_embed = brainbert_processor(seeg)
        seeg_embed = seeg_embed.reshape(batch_shape, electrode_shape, seeg_embed.shape[1], seeg_embed.shape[2])
        seeg_embed = seeg_encoder(seeg_embed)
        optimizer.zero_grad()
        loss = loss_func(img_embed, seeg_embed)
        loss.backward()
        train_loss_list.append(loss.item())
        optimizer.step()

        if step % 100 == 0:
            print(np.mean(train_loss_list))
            train_loss_list = []
    return np.mean(train_loss_list)


def val_fn(seeg_encoder, loss_func, clip_processor, brainbert_processor, val_loader):
    seeg_encoder.eval()
    val_loss_list = []
    with torch.no_grad():
        for index, (seeg, img, _, _, _, _) in enumerate(val_loader):
            batch_shape, electrode_shape = seeg.shape[0], seeg.shape[1]
            seeg, img = seeg.reshape(-1, seeg.shape[-1]).to(device), img.to(device).half()
            img_embed = clip_processor(img)
            seeg_embed = brainbert_processor(seeg)
            seeg_embed = seeg_embed.reshape(batch_shape, electrode_shape, seeg_embed.shape[1], seeg_embed.shape[2])
            seeg_embed = seeg_encoder(seeg_embed)
            loss = loss_func(img_embed, seeg_embed)
            val_loss_list.append(loss.item())
    return np.mean(val_loss_list)


def train(params):
    dataset = SeegDataset('../data/paired_data/sub_07_data.h5')
    train_loader, val_loader = create_dataloaders(dataset=dataset, batch_size=params['batch'])

    seeg_encoder = SeegEncoder().to(device)
    clip_processor = ClipProcessor().to(device)
    brainbert_processor = BrainBERTProcessor().to(device)

    optimizer = optim.AdamW(
        params=seeg_encoder.parameters(),
        lr=params['lr'],
        betas=(params['beta1'], params['beta2']),
        eps=params['eps'],
    )

    loss_func = partial(loss_fn)

    # Record loss
    training_loss_list = []
    val_loss_list = []
    for i in range(params['epoch']):
        train_loss = train_fn(
            seeg_encoder=seeg_encoder,
            optimizer=optimizer,
            clip_processor=clip_processor,
            brainbert_processor=brainbert_processor,
            loss_func=loss_func,
            train_loader=train_loader,
        )
        val_loss = val_fn(
            seeg_encoder=seeg_encoder,
            loss_func=loss_func,
            clip_processor=clip_processor,
            brainbert_processor=brainbert_processor,
            val_loader=val_loader,
        )
        print("Epoch", i)
        print(f"train_loss:{train_loss.item()}")
        print(f"val_loss:{val_loss.item()}")
        training_loss_list.append(train_loss.item())
        val_loss_list.append(val_loss.item())
        torch.save(seeg_encoder.state_dict(), f'checkpoint{i}.pth')


def main():
    params = {
        'epoch': 10,
        'batch': 8,
        "lr": 1.0e-6,
        "beta1": 0.9,
        "beta2": 0.98,
        "eps": 1.0e-6
    }
    train(params)


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    set_seed(42)
    main()
