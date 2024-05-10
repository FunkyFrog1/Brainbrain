import torch
from tqdm import tqdm
from dataloader import create_dataloaders, SeegDataset
from clip_process import clip_process
from brainbert_process import brainbert_process
from seeg_encoder import SeegEncoder


def criterion():
    pass


def train_fn():
    pass


def val_fn():
    pass


def train():
    pass


def main():
    dataset = SeegDataset('../data/paired_data/sub_07_data.h5')
    train_loader, val_loader = create_dataloaders(dataset=dataset, batch_size=2)
    seeg_encoder = SeegEncoder().to(device)

    for index, (seeg, img, _, _, _, _) in enumerate(tqdm(train_loader)):
        batch_shape, electrode_shape = seeg.shape[0], seeg.shape[1]
        seeg, img = seeg.reshape(-1, seeg.shape[-1]).to(device), img.to(device).half()

        seeg_embed = brainbert_process(seeg)
        seeg_embed = seeg_embed.reshape(batch_shape, electrode_shape, seeg_embed.shape[1], seeg_embed.shape[2])
        img_embed = clip_process(img)

        output = seeg_encoder(seeg_embed)
        print(output.shape)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    main()
