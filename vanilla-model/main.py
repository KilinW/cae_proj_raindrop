import torch
from model import voltageNN
from dataset import voltageDataset

model = voltageNN()
dataloader = torch.utils.data.DataLoader(voltageDataset("/home/aicenter/cae_proj_raindrop/data/voltage"),
                                         shuffle = True,
                                         batch_size = 32,
                                         drop_last = True)
epoch = 5000

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                             lr = 0.0001)

for e in range(epoch):
    for idx, (x, y) in enumerate(dataloader):
        pred = model(x)
        
        loss = loss_fn(pred, y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()