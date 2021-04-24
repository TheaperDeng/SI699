import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tcn import TemporalConvNet

# train_x = np.load('train_x_coor.npy', allow_pickle=True)
# train_y = np.load('train_y_coor.npy', allow_pickle=True)
# valid_x = np.load('valid_x_coor.npy', allow_pickle=True)
# valid_y = np.load('valid_y_coor.npy', allow_pickle=True)


train_x = np.load('train_x_ele_24.npy')
train_y = np.load('train_y_ele_24.npy')
valid_x = np.load('valid_x_ele_24.npy')
valid_y = np.load('valid_y_ele_24.npy')

print(train_x.shape, train_y.shape)
print(valid_x.shape, valid_y.shape)

model = TemporalConvNet(past_seq_len=36,
                        input_feature_num=2,
                        future_seq_len=12,
                        output_feature_num=1,
                        num_channels=[16] * 6,
                        kernel_size=3,
                        dropout=0.2).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

def train_epoch(model, x, y, opt, loss_fn, x_test, y_test):
    batch_size = 512
    model.train()
    total_loss = 0
    train_loader = DataLoader(TensorDataset(x, y),
                                batch_size=int(batch_size),
                                shuffle=True)
    batch_idx = 0
    for x_batch, y_batch in train_loader:
        opt.zero_grad()
        yhat = model(x_batch)
        loss = loss_fn(yhat, y_batch)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        batch_idx += 1
        if np.isnan(loss.item()):
            print(x_batch, y_batch)
            break
        if batch_idx % 100 == 1:
            print(f"train loss {batch_idx}/{len(train_loader)}:", loss.item())
    model.eval()
    yhat = model(x_test)
    loss = loss_fn(yhat, y_test)
    print("test loss:", loss.item())

# for i in range(30):
#     train_epoch(model,
#                 torch.from_numpy(train_x.astype(np.float32).reshape(train_x.shape[0], train_x.shape[1], 1)).cuda(), 
#                 torch.from_numpy(train_y.astype(np.float32).reshape(train_y.shape[0], train_y.shape[1], 1)).cuda(), 
#                 opt, 
#                 loss_fn,
#                 torch.from_numpy(valid_x.astype(np.float32).reshape(valid_x.shape[0], valid_x.shape[1], 1)).cuda(),
#                 torch.from_numpy(valid_y.astype(np.float32).reshape(valid_y.shape[0], valid_y.shape[1], 1)).cuda(),)

for i in range(40):
    train_epoch(model,
                torch.from_numpy(train_x[:,:,:]).cuda(), 
                torch.from_numpy(train_y[:,:,:]).cuda(), 
                opt, 
                loss_fn,
                torch.from_numpy(valid_x[:,:,:]).cuda(),
                torch.from_numpy(valid_y[:,:,:]).cuda(),)
