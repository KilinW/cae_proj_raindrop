import torch
from torch.utils.data import Dataset
import os
import numpy as np

class voltageDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.data = []
        self.ans = []
        for file in os.listdir(self.dir):
            if file.endswith(".npy"):
                self.data.append(np.load(os.path.join(self.dir, file)))
                self.ans.append(float(os.path.splitext(file)[0].split("_")[1]))
    
    def __getitem__(self, index):
        # trim as same size, since the npy file some are 1005, some are 1004
        return (self.data[index][:1000]), (self.ans[index])


    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    generator = torch.Generator().manual_seed(42)
    [train, val, test] = torch.utils.data.random_split(voltageDataset("/home/aicenter/cae_proj_raindrop/new-data/voltage"),
                                                     [3396, 425, 425],
                                                     generator)
    print(len(train), len(val), len(test))
    train_dataloader = torch.utils.data.DataLoader(train,
                                               shuffle = True,
                                               batch_size = 32,
                                               drop_last = True)
    val_dataloader = torch.utils.data.DataLoader(val,
                                                shuffle = True,
                                                batch_size = 32,
                                                drop_last = True)
    test_dataloader = torch.utils.data.DataLoader(test,
                                                shuffle = True,
                                                batch_size = 32,
                                                drop_last = True)
    for idx, batch in enumerate(train_dataloader):
        if idx == 1:
            break
        print(type(batch))
        # print(batch.shape)
        print(batch)
        print(idx)