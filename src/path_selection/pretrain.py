from net import *
import torch.nn as nn
import torch
import numpy as np
import json
from configparser import ConfigParser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, help="the config file", default="./config/pretrain.config"
)
args = parser.parse_args()

config = ConfigParser()
config.read(args.config)

# Load Data
train_data = np.load(config["path"]["data_path"] + "1fea1.npy")
feature = torch.Tensor(train_data)

# Inti Net
net = AutoEncoder(1, 1, json.loads(config.get("trainsetting", "layers")))
if torch.cuda.is_available():
    net = net.cuda()
    feature = feature.cuda()
net.train()

cri = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=config.getfloat("trainsetting", "lr"))
batch_size = config.getint("trainsetting", "batch_size")
n = len(feature)

# Train
for epoch in range(1, config.getint("trainsetting", "epoch") + 1):

    opt.zero_grad()
    output = net(feature, c_state=False)
    loss = cri(output["decode"], feature)

    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        print("epoch:", epoch, "loss:", loss.item())

torch.save(net.state_dict(), config["path"]["model_save_path"] + "/only_AE.pt")
