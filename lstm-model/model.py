import torch
from dataset import voltageDataset

class voltageNN(torch.nn.Module):
    def __init__(self):
        super(voltageNN, self).__init__()
        self.lstm = torch.nn.LSTM(input_size = 1,
                                  hidden_size = 256,
                                  num_layers = 2,
                                  bias = True,
                                  dropout = 0,
                                  batch_first = True,
                                  bidirectional = False,
                                  proj_size = 1, 
                                  dtype=torch.float64)
        self.linear1 = torch.nn.Linear(1000, 100, dtype=torch.float64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 1, dtype=torch.float64)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.linear1(torch.squeeze(output))
        output = self.relu(output)
        output = self.linear2(output)
        return output
    
model = voltageNN()
print(model)
dataloader = torch.utils.data.DataLoader(voltageDataset("/home/aicenter/cae_proj_raindrop/data/voltage"),
                                         shuffle = True,
                                         batch_size = 2,
                                         drop_last = True)
for idx, (x, y) in enumerate(dataloader):
    if idx == 1:
        break
    # print(x)
    # print(x.unsqueeze(2))
    # print(x.unsqueeze(2).shape)
    pred = model(x.unsqueeze(2))
    print(pred)
    print(y)
