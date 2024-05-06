import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class SeegDataset(Dataset):
    def __init__(self, archive, seeg='seeg', first_frame='first_frame', movie_num='movie_num', clip='clip',
                 clip_sub='clip_sub', watch_flag='watch_flag', train=True, split_ratio=0.9):
        self.archive = h5py.File(archive, 'r')
        self.seeg = self.archive[seeg]
        self.first_frame = self.archive[first_frame]
        self.movie_num = self.archive[movie_num]
        self.clip = self.archive[clip]
        self.clip_sub = self.archive[clip_sub]
        self.watch_flag = self.archive[watch_flag]

        indices = list(range(len(self.seeg)))
        train_indices, test_indices = train_test_split(indices, test_size=1-split_ratio, random_state=42)
        self.indices = train_indices if train else test_indices

    def __getitem__(self, index):
        actual_index = self.indices[index]
        seeg = self.seeg[actual_index]
        first_frame = self.first_frame[actual_index]
        movie_num = self.movie_num[actual_index]
        clip = self.clip[actual_index]
        clip_sub = self.clip_sub[actual_index]
        watch_flag = self.watch_flag[actual_index]
        return seeg, first_frame, movie_num, clip, clip_sub, watch_flag

    def __len__(self):
        return len(self.indices)

    def close(self):
        self.archive.close()



def test():
    # 使用这个类
    train_data = SeegDataset('../data/sub_07_data.h5', train=True)
    train_loader = DataLoader(dataset=train_data, num_workers=0, batch_size=100, shuffle=True)

    # 使用这个类
    test_data = SeegDataset('../data/sub_07_data.h5', train=False)
    test_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=100, shuffle=True)

    print(len(train_loader.dataset))
    print(len(test_loader.dataset))

    # # 遍历DataLoader
    # for batch in tqdm(train_loader):
    #     seeg, first_frame, movie_num, clip, clip_sub, watch_flag = batch
    #     print(seeg.shape)
    #     print(first_frame.shape)


if __name__ == '__main__':
    test()
