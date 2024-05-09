import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch


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

    # def preprocess_depth(self, depth, batch_size=64):
    #     # 检查是否有可用的GPU
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     # 预处理数据
    #     preprocessed_depth = None
    #     depth_extractor = ().to(device)
    #     depth_extractor.load_state_dict(torch.load(os.path.join(self.current_dir, "../Depth_AE_aug/Depth_AutoEncoder.pth"), map_location=device))
    #     depth_extractor.eval()
    #
    #     depth = depth.to(device)
    #     with torch.no_grad():
    #         for i in range(0, len(depth), batch_size):
    #             batch = depth[i:i + batch_size]
    #             batch = depth_extractor.encoder(batch.unsqueeze(1)).squeeze()
    #             if preprocessed_depth is None:
    #                 preprocessed_depth = batch.cpu()
    #             else:
    #                 preprocessed_depth = torch.cat((preprocessed_depth, batch.cpu()), dim=0)
    #
    #     # 清理缓存
    #     del depth_extractor
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #
    #     return preprocessed_depth

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
    train_data = SeegDataset('../data/paired_data/sub_07_data.h5', train=True)
    train_loader = DataLoader(dataset=train_data, num_workers=0, batch_size=2, shuffle=True)

    # 使用这个类
    test_data = SeegDataset('../data/paired_data/sub_07_data.h5', train=False)
    test_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=2, shuffle=True)

    print(len(train_loader.dataset))
    print(len(test_loader.dataset))

    # # 遍历DataLoader
    for batch in tqdm(train_loader):
        seeg, first_frame, movie_num, clip, clip_sub, watch_flag = batch
        print(seeg.shape)
        print(first_frame.shape)
        print(movie_num)
        print(clip)
        print(clip_sub)
        print(watch_flag)

        break


if __name__ == '__main__':
    test()
