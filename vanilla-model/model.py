import torch
from dataset import voltageDataset

class voltageNN(torch.nn.Module):
    def __init__(self):
        super(voltageNN, self).__init__()
        self.input = torch.nn.Linear(1000, 512, dtype=torch.float64)
        self.linear1 = torch.nn.Linear(512, 256, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(256, 256, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(256, 256, dtype=torch.float64)
        self.linear4 = torch.nn.Linear(256, 256, dtype=torch.float64)
        self.linear8 = torch.nn.Linear(256, 128, dtype=torch.float64)
        self.relu1 = torch.nn.ReLU()
        self.output = torch.nn.Linear(128, 1, dtype=torch.float64)
    
    def forward(self, x):
        
        output = self.input(x)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        output = self.linear8(output)
        output = self.output(output)

        return output
    

if __name__ == "__main__":
    model = voltageNN()
    print(model)
    dataloader = torch.utils.data.DataLoader(voltageDataset("/home/aicenter/cae_proj_raindrop/data/voltage"),
                                            shuffle = True,
                                            batch_size = 1,
                                            drop_last = True)
    for idx, (x, y) in enumerate(dataloader):
        if idx == 1:
            break
        res = model(x)
        print(res)
        print(y)
