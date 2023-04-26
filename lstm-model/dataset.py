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
                self.ans.append(float(os.path.splitext(file)[0][8:]))
    
    def __getitem__(self, index):
        # trim as same size, since the npy file some are 1005, some are 1004
        return (self.data[index][:1000]), (self.ans[index])


    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dataloader = torch.utils.data.DataLoader(voltageDataset("/home/aicenter/cae_proj_raindrop/data/voltage"),
                                            shuffle = True,
                                            batch_size = 20,
                                            drop_last = False)
    for idx, batch in enumerate(dataloader):
        print(type(batch))
        # print(batch.shape)
        print(batch)
        print(idx)