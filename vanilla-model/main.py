import torch
from model import voltageNN
from dataset import voltageDataset
import json

model = voltageNN()
generator = torch.Generator().manual_seed(42)
[train, val, test] = torch.utils.data.random_split(voltageDataset("/home/aicenter/cae_proj_raindrop/new-data/voltage"),
                                                    [3396, 425, 425],
                                                    generator)
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
epoch = 200

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                             lr = 0.0005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
history = {}
history["train"] = []
history["val"] = []
history["test"] = []

curr_loss = 2**16
for e in range(epoch):
    totalLoss = []
    model.train()
    for idx, (x, y) in enumerate(train_dataloader):
        pred = model(x)
        
        loss = loss_fn(pred, y.unsqueeze(1))
        # print(loss)
        totalLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch: {e}, train loss: {sum(totalLoss)/len(totalLoss)}")
    history["train"].append(sum(totalLoss)/len(totalLoss))
    scheduler.step()
    if (e+1) % 100 == 0:
        torch.save(model.state_dict(), f"/home/aicenter/cae_proj_raindrop/vanilla-model/result/epoch{e+1}_model.pth")

    model.eval()
    with torch.no_grad():
        totalLoss = []
        for idx, (x, y) in enumerate(val_dataloader):
            pred = model(x)
            loss = loss_fn(pred, y.unsqueeze(1))
            totalLoss.append(loss.item())
        print(f"epoch: {e}, val loss: {sum(totalLoss)/len(totalLoss)}")
        history["val"].append(sum(totalLoss)/len(totalLoss))

        if curr_loss > sum(totalLoss)/len(totalLoss):
            curr_loss = sum(totalLoss)/len(totalLoss)
            torch.save(model.state_dict(), "/home/aicenter/cae_proj_raindrop/vanilla-model/new-result/best_model.pth")

    model.eval()
    with torch.no_grad():
        totalLoss = []
        for idx, (x, y) in enumerate(test_dataloader):
            pred = model(x)
            loss = loss_fn(pred, y.unsqueeze(1))
            totalLoss.append(loss.item())
        print(f"epoch: {e}, test loss: {sum(totalLoss)/len(totalLoss)}")
        history["test"].append(sum(totalLoss)/len(totalLoss))

    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)
    