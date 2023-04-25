import torch
from dataset import voltageDataset

class voltageNN(torch.nn.Module):
    def __init__(self):
        super(voltageNN, self).__init__()
        self.input = torch.nn.Linear(1000, 100, dtype=torch.float64)
        self.relu1 = torch.nn.ReLU()
        self.output = torch.nn.Linear(100, 1, dtype=torch.float64)
    
    def forward(self, x):
        output = self.input(x)
        output = self.relu1(output)
        output = self.output(output)
        return output
    
class voltageLSTM(torch.nn.Module):
    def __init__(self):
        super(voltageLSTM, self).__init__()
        self.lstm = torch.nn.LSTM()
model = voltageNN()
print(model)
dataloader = torch.utils.data.DataLoader(voltageDataset("/home/aicenter/cae_proj_raindrop/data/voltage"),
                                         shuffle = True,
                                         batch_size = 20,
                                         drop_last = True)
for idx, (x, y) in enumerate(dataloader):
    print(idx)
    res = model(x)

    print(res-y)