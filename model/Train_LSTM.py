import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

# train_x = np.load('train_x_coor.npy', allow_pickle=True)
# train_y = np.load('train_y_coor.npy', allow_pickle=True)
# valid_x = np.load('valid_x_coor.npy', allow_pickle=True)
# valid_y = np.load('valid_y_coor.npy', allow_pickle=True)

train_x = np.load('train_x_ele.npy')
train_y = np.load('train_y_ele.npy')
valid_x = np.load('valid_x_ele.npy')
valid_y = np.load('valid_y_ele.npy')

# train_x_scale = standard_scaler.fit_transform(train_x)
# train_y_scale = standard_scaler.transform(train_y)
# valid_x_scale = standard_scaler.transform(valid_x)
# valid_y_scale = standard_scaler.transform(valid_y)

print(train_x.shape, train_y.shape)
print(valid_x.shape, valid_y.shape)

class LSTMForecaster(nn.Module):
    def __init__(self, input_feature_num, output_length):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_feature_num, 64, 4, dropout=0, batch_first=True)
        self.fc = nn.Linear(64, output_length)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        out = self.fc(lstm_out[:, -1, :])
        return out

model = LSTMForecaster(input_feature_num=2, output_length=12).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

def train_epoch(model, x, y, opt, loss_fn, x_test, y_test):
    batch_size = 1024
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
        # if np.isnan(loss.item()):
            # print(x_batch, y_batch)
            # break
        if batch_idx % 100 == 1:
            print(f"train loss {batch_idx}/{len(train_loader)}:", loss.item())
    model.eval()
    yhat = model(x_test)
    loss = loss_fn(yhat, y_test)
    print(np.mean((yhat.cpu().detach().numpy() - y_test.cpu().detach().numpy())**2, axis=0))
    print("test loss:", loss.item())

# for i in range(30):
#     train_epoch(model,
#                 torch.from_numpy(train_x.astype(np.float32).reshape(train_x.shape[0], train_x.shape[1], 1)).cuda(), 
#                 torch.from_numpy(train_y[:,:].astype(np.float32)).cuda(), 
#                 opt, 
#                 loss_fn,
#                 torch.from_numpy(valid_x.astype(np.float32).reshape(valid_x.shape[0], valid_x.shape[1], 1)).cuda(),
#                 torch.from_numpy(valid_y[:,:].astype(np.float32)).cuda(),)

for i in range(40):
    train_epoch(model,
                torch.from_numpy(train_x[:,:,:]).cuda(), 
                torch.from_numpy(train_y[:,:,0]).cuda(), 
                opt, 
                loss_fn,
                torch.from_numpy(valid_x[:,:,:]).cuda(),
                torch.from_numpy(valid_y[:,:,0]).cuda(),)
