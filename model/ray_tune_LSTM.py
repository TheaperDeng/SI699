import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
import pandas as pd

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, FIFOScheduler

import torch
import torch.nn as nn



class LSTMForecaster(nn.Module):
    def __init__(self, input_feature_num, output_length, hiddensize, level):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_feature_num, hiddensize, level, dropout=0, batch_first=True)
        self.fc = nn.Linear(hiddensize, output_length)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        out = self.fc(lstm_out[:, -1, :])
        return out


def train(model, optimizer, train_loader, device=torch.device("cpu")):
    print("Start training this model")
    loss_fn = torch.nn.MSELoss()
    model.train()
    for epoch in range(0, EPOCH_NUM):
        batch_idx = 0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = model(x_batch)
            loss = loss_fn(yhat, y_batch)
            loss.backward()
            optimizer.step()
            batch_idx += 1
            if batch_idx % 100 == 1:
                print(f"train loss {batch_idx}/{len(train_loader)}:", loss.item())

def test(model, x_test, y_test, device=torch.device("cpu")):
    loss_fn = torch.nn.MSELoss()
    model.eval()
    yhat = model(x_test)
    loss = loss_fn(yhat, y_test)
    print("test loss:", loss.item())
    return loss.item()

def get_data(config):
    # train_x = np.load('C:\\Users\\Theap\\Documents\\SI699\\experimental_baseline\\train_x_coor.npy', allow_pickle=True)
    # train_y = np.load('C:\\Users\\Theap\\Documents\\SI699\\experimental_baseline\\train_y_coor.npy', allow_pickle=True)
    # valid_x = np.load('C:\\Users\\Theap\\Documents\\SI699\\experimental_baseline\\valid_x_coor.npy', allow_pickle=True)
    # valid_y = np.load('C:\\Users\\Theap\\Documents\\SI699\\experimental_baseline\\valid_y_coor.npy', allow_pickle=True)

    train_x = np.load('C:\\Users\\Theap\\Documents\\SI699\\experimental_baseline\\train_x_ele.npy')
    train_y = np.load('C:\\Users\\Theap\\Documents\\SI699\\experimental_baseline\\train_y_ele.npy')
    valid_x = np.load('C:\\Users\\Theap\\Documents\\SI699\\experimental_baseline\\valid_x_ele.npy')
    valid_y = np.load('C:\\Users\\Theap\\Documents\\SI699\\experimental_baseline\\valid_y_ele.npy')
    train_x = torch.from_numpy(train_x[:,:,:]).cuda()
    train_y = torch.from_numpy(train_y[:,:,0]).cuda()
    valid_x = torch.from_numpy(valid_x[:,:,:]).cuda()
    valid_y = torch.from_numpy(valid_y[:,:,0]).cuda()
    # train_x = torch.from_numpy(train_x.astype(np.float32).reshape(train_x.shape[0], train_x.shape[1], 1)).cuda()
    # train_y = torch.from_numpy(train_y[:,:].astype(np.float32)).cuda()
    # valid_x = torch.from_numpy(valid_x.astype(np.float32).reshape(valid_x.shape[0], valid_x.shape[1], 1)).cuda()
    # valid_y = torch.from_numpy(valid_y[:,:].astype(np.float32)).cuda()
    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=config['batch_size'],
                              shuffle=True)
    return train_loader, valid_x, valid_y

def train_LSTM(config):
    train_loader, valid_x, valid_y = get_data(config)
    model = LSTMForecaster(input_feature_num=2, output_length=12, hiddensize=config['hiddensize'], level=config['level']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    min_mse = 999
    while True:
        import time
        st = time.time()
        train(model, optimizer, train_loader)
        train_end = time.time()
        print(f"train takes {train_end - st}")
        mse = test(model, valid_x, valid_y)
        min_mse = min(min_mse, mse)
        tune.report(MSE=min_mse)

if __name__ == "__main__":
    torch.set_num_threads(2)
    ray.init(num_cpus=4)
    EPOCH_NUM = 1

    # for early stopping
    # sched = AsyncHyperBandScheduler()
    sched = FIFOScheduler()
    smoke_test = False

    analysis = tune.run(
        train_LSTM,
        name="LSTM_smoke_{}_country_with_extra_{}".format(smoke_test, time.time()),
        mode="min",
        metric="MSE",
        scheduler=sched,
        stop={
            "training_iteration": 1 if smoke_test else 30 
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        num_samples= 1,
        config={
            "lr": tune.choice([0.003]),
            "batch_size": tune.grid_search([512]),
            "hiddensize": tune.grid_search([16]),
            "level": tune.grid_search([2, 4]),
        } if smoke_test else {
            "lr": tune.grid_search([0.001, 0.005]),
            "batch_size": tune.grid_search([512]),
            "hiddensize": tune.grid_search([32, 64]),
            "level": tune.grid_search([2, 3, 4]),
        })
    print("Best config is:", analysis.best_config)