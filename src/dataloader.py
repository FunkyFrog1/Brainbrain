import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
from tqdm import tqdm


class SeegDataset(Dataset):
    def __init__(self, archive, seeg='seeg', first_frame='first_frame', movie_num='movie_num', clip='clip',
                 clip_sub='clip_sub', watch_flag='watch_flag', split_ratio=0.9):
        self.archive = h5py.File(archive, 'r')
        self.seeg = self.archive[seeg]
        self.first_frame = self.archive[first_frame]
        self.movie_num = self.archive[movie_num]
        self.clip = self.archive[clip]
        self.clip_sub = self.archive[clip_sub]
        self.watch_flag = self.archive[watch_flag]

        self.indices = list(range(len(self.seeg)))
        self.train_indices, self.test_indices = train_test_split(self.indices, test_size=200,
                                                                 random_state=42)

    def __getitem__(self, index):
        seeg = self.seeg[index]
        first_frame = self.first_frame[index]
        movie_num = self.movie_num[index]
        clip = self.clip[index]
        clip_sub = self.clip_sub[index]
        watch_flag = self.watch_flag[index]
        return seeg, first_frame, movie_num, clip, clip_sub, watch_flag

    def __len__(self):
        return len(self.indices)

    def close(self):
        self.archive.close()


def create_dataloaders(dataset, train_batch=32, val_batch=20, train_shuffle=True, val_shuffle=False):
    train_indices = dataset.train_indices
    val_indices = dataset.test_indices

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SequentialSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=train_batch, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=val_batch, sampler=val_sampler)

    return train_loader, val_loader


def test():
    dataset = SeegDataset('../data/paired_data/sub_07_data.h5')
    train_loader, val_loader = create_dataloaders(dataset=dataset, batch_size=1)

    for batch in tqdm(train_loader):
        pass
    for batch in tqdm(val_loader):
        pass


if __name__ == '__main__':
    test()
