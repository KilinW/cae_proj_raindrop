import torch
from model import voltageNN
from dataset import voltageDataset


model = voltageNN().to("cuda")
dataloader = torch.utils.data.DataLoader(voltageDataset("/home/aicenter/cae_proj_raindrop/data/voltage"),
                                         shuffle = True,
                                         batch_size = 32,
                                         drop_last = True)
epoch = 1000

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 0.0005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

curr_loss = 100
for e in range(epoch):
    totalLoss = []
    for idx, (x, y) in enumerate(dataloader):
        x, y = x.to("cuda"), y.to("cuda")
        pred = model(x.unsqueeze(2))
        
        loss = loss_fn(pred, y)
        # print(loss)
        totalLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch: {e}, loss: {sum(totalLoss)/len(totalLoss)}")
    
    if curr_loss > sum(totalLoss)/len(totalLoss):
        curr_loss = sum(totalLoss)/len(totalLoss)
        torch.save(model.state_dict(), "/home/aicenter/cae_proj_raindrop/lstm-model/result/best_model.pth")
    
    if (e+1) % 100 == 0:
        torch.save(model.state_dict(), f"/home/aicenter/cae_proj_raindrop/lstm-model/result/epoch{e+1}_model.pth")

    scheduler.step()