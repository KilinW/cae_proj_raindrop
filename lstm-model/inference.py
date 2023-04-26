import torch
import os
import numpy as np
from model import voltageNN
from dataset import voltageDataset

model = voltageNN()

model.load_state_dict(torch.load("/home/aicenter/cae_proj_raindrop/lstm-model/result/epoch100_model.pth"))
model.eval()

input_file = "voltage_2.88.npy"
input_path = os.path.join("/home/aicenter/cae_proj_raindrop/data/voltage", input_file)
input = torch.tensor(np.load(input_path)[:1000])
output = float(os.path.splitext(input_file)[0][8:])

# print(input.unsqueeze(0).unsqueeze(2).shape)
pred = model(input.unsqueeze(0).unsqueeze(2))


print(f"pred: {pred}, ans: {output}")
