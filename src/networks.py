#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
networks for experiments
"""

import torch
import torch.nn as nn

class LayerNet(nn.Module):
    def __init__(self, nx, nh, ny, act = torch.sigmoid, d = 2):
        ''' neural network with d hidden layers with the same size
        Parameters
        --------
        nx (int):
            number of input nodes
        nh (int):
            number of hidden nodes per layer
        ny (int):
            number of output nodes
        act (torch.function):
            activation function per layer
        d (int):
            number of hidden layers
        '''
        super(LayerNet, self).__init__()
        # an affine operation: y = Wx + b
        self.act = act
        if d > 0:
            self.linears = nn.ModuleList([nn.Linear(nx,nh)] + [nn.Linear(nh, nh) for i in range(d-1)])
            self.fc2 = nn.Linear(nh, ny)
        else:
            self.linears = []
            self.fc2 = nn.Linear(nx,ny)
        
    def forward(self, x):
        x = x.flatten(1)
        for i, l in enumerate(self.linears):
            x = self.act(l(x))
            if i == 0:
                self.z = x
        self.y = self.fc2(x)
        return self.y
    
    
