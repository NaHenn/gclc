#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implementation of an example how to use optim_GCLC and compare different methods
"""
import torch
import time
import pandas as pd

from optim_GCLC import GCLC
from networks import LayerNet

def run_GCLC(optimizer, x, y, model, loss_fn, cycles, verbose = 0):
    def closure():
        with torch.enable_grad():
            optimizer.zero_grad()    
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)
            # Compute loss.
            loss = loss_fn(y_pred, y)
            loss.backward(retain_graph = True)
            return loss
    
    times = []
    error_codes = []
    for t in range(cycles):   
        st = time.perf_counter()
        loss = optimizer.step(closure)
        times.append(time.perf_counter() - st)
        if verbose and t % 10 == 9:
            print(t, loss)
        if t == 0:
            loss_values = optimizer.state[optimizer._params[0]]['all_loss']
        else:
            loss_values += optimizer.state[optimizer._params[0]]['all_loss'][1:]
        error_codes.append(loss[1])
        if verbose and optimizer.state[optimizer._params[0]]['error_code'] != 0:
            print(t, optimizer.state[optimizer._params[0]]['error_code'])
    return {'loss': loss_values, 'time': times, 'error_codes': error_codes}

def run_GD(optimizer, x, y, model, loss_fn, cycles, verbose = 0):
    def closure():
        with torch.enable_grad():
            optimizer.zero_grad()    
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)
            # Compute loss.
            loss = loss_fn(y_pred, y)
            loss.backward(retain_graph = True)
            return loss
    
    times = []
    for t in range(cycles):   
        st = time.perf_counter()
        loss = optimizer.step(closure)
        times.append(time.perf_counter() - st)
        if verbose and t % 10 == 9:
            print(t, loss)
        if t == 0:
            loss_values = [loss.item()]
        else:
            loss_values += [loss.item()]
    return {'loss': loss_values, 'time': times}

# load data
data = pd.read_csv('../datasets/smith/concrete.dat', sep = ',', header = None)
x = torch.tensor(data.drop(data.columns[-1],axis=1).to_numpy(), dtype=torch.float32)
y = torch.tensor(data[data.columns[-1]], dtype=torch.float32).unsqueeze(1)

# scale data
x = 2* (x - x.min(dim=0)[0])/(x.max(dim=0)[0] - x.min(dim=0)[0]) - 1
y = 2* (y - y.min(dim=0)[0])/(y.max(dim=0)[0] - y.min(dim=0)[0]) - 1


# initialize network, save weights for the other runs
model = LayerNet(x.shape[-1], 10, 1, d = 1)
save_weights = torch.nn.utils.parameters_to_vector(model.parameters()).clone()

loss_fn = torch.nn.MSELoss()
learning_rate = 0.1
cycles = 10

# optimizers
opt = torch.optim.SGD(model.parameters(), lr = learning_rate)
GCLC00 = GCLC(model.parameters(), opt = opt, lr=learning_rate, pre_iter = 3,post_iter = 1, m = 0, l = 0, line_search_fn="wolfe12")
GCLC10 = GCLC(model.parameters(), opt = opt, lr=learning_rate, pre_iter = 3,post_iter = 1, m = 0, l = 1, line_search_fn="wolfe12")
GCLC11 = GCLC(model.parameters(), opt = opt, lr=learning_rate, pre_iter = 3,post_iter = 1, m = 1, l = 1, line_search_fn="wolfe12")
GCLC44 = GCLC(model.parameters(), opt = opt, lr=learning_rate, pre_iter = 3,post_iter = 1, m = 4, l = 4, line_search_fn="wolfe12")

exp_dict = {'GCLC(0,0)': {'optimizer': GCLC00},'GCLC(1,0)': {'optimizer': GCLC10},
           'GCLC(1,1)': {'optimizer': GCLC11},'GCLC(4,4)': {'optimizer': GCLC44}}
for exp_name, dct in exp_dict.items():
    # ensure the same starting point for all experiments
    torch.nn.utils.vector_to_parameters(save_weights.clone(), model.parameters())
    dct.update(run_GCLC(dct['optimizer'], x, y, model, loss_fn, cycles))
    
# comparison with GD
torch.nn.utils.vector_to_parameters(save_weights.clone(), model.parameters())
exp_dict.update({'GD': {'optimizer': opt}})
exp_dict['GD'].update(run_GD(opt, x,y,model, loss_fn, 5*cycles))

print("loss at the end:")
for exp_name, dct in exp_dict.items():
    print('{:10} loss: {}'.format(exp_name, dct['loss'][-1]))

