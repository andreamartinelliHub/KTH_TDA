import numpy as np
import pandas as pd
from sklearn.preprocessing  import LabelEncoder, OneHotEncoder

import random
import itertools
import functools
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def rand(shape, low, high):
    """Tensor of random numbers, uniformly distributed on [low, high]."""
    return torch.rand(shape) * (high - low) + low


class DeepSetLayer(nn.Module):
    """
    DeepSetLayer(in_blocks, out_blocks) takes shape (batch, in_blocks, n) to (batch, out_blocks, n).
    Each block of n scalars is treated as the S_n permutation representation, and maps between blocks are
    S_n-equivariant.
    """
    def __init__(self, in_blocks, out_blocks, apply_reduction=True, reduction="sum"):
        super().__init__()
        
        self.in_blocks = in_blocks
        self.out_blocks = out_blocks
        self.apply_reduction = apply_reduction
        self.reduction = reduction
        
        # Initialisation tactic copied from nn.Linear in PyTorch
        #lim = (in_blocks)**-0.5 / 2
        lim = (in_blocks*1000)**-0.5 / 2

        # Alpha corresponds to the identity, beta to the all-ones matrix, and gamma to the additive bias.
        self.alpha = torch.nn.Parameter(data=rand((out_blocks, in_blocks), -lim, lim))
        self.gamma = torch.nn.Parameter(data=rand((out_blocks), -lim, lim))
        
        if apply_reduction:
            self.beta = torch.nn.Parameter(data=rand((out_blocks, in_blocks), -lim, lim))
            
    
    def forward(self, x):
        # x has shape (batch, in_blocks, n)
        
        res = torch.einsum('...jz, ij -> ...iz', x, self.alpha)
        + self.gamma[..., None]
        
        if self.apply_reduction:
            if self.reduction=="max":
                res += torch.einsum('...jz, ij -> ...iz', x.max(axis=-1)[0][..., None], self.beta)
            else: #default is sum
                res += torch.einsum('...jz, ij -> ...iz', x.sum(axis=-1)[..., None], self.beta)
            
        
        return res

        
class DeepSetTopK(nn.Module):
    """
    DeepSetSum(blocks) takes a deep set layer of shape (batch, blocks, n) to a regular layer
    of shape (batch, blocks) by projecting to the trivial representation and then extracting
    a coordinate, eg
        (1, 2, 3, 4) => project to trivial => (2.5, 2.5, 2.5, 2.5) => extract component => 2.5
    """
    def __init__(self, k):
        super().__init__()
        
        self.k = k
        
    
    def forward(self, x):
        return torch.topk(x, self.k, axis=2)[0].reshape(x.shape[0], -1)
  



class Model_DeepSet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.h0 = nn.Sequential(
            DeepSetLayer(2, 25, apply_reduction=False),
            DeepSetLayer(25, 25, apply_reduction=True, reduction="max"),
            DeepSetTopK(5)
        )

        self.h1 = nn.Sequential(
            DeepSetLayer(2, 25, apply_reduction=False),
            DeepSetLayer(25, 25, apply_reduction=True, reduction="max"),
            DeepSetTopK(5)
        )

        self.merge = nn.Sequential(
            nn.Linear(2*25*5, 5),
            #nn.Sigmoid()
        )
        
        
    def forward(self, x):
        h0_rep = self.h0(x[0])
        h1_rep = self.h1(x[1])
        
        reps = torch.concatenate((h0_rep, h1_rep), axis=1)
        
        logits = self.merge(reps)
        
        return logits
    


def train(dataloader, model, loss_fn, optimizer, device, printit):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = [x.to(device) for x in X], y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.argmax(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if printit==True:
          if batch % 100 == 0:
              loss, current = loss.item(), (batch + 1) * len(X)
              print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    preds = []
    ground_truth = []
    with torch.no_grad():
        for X, y in dataloader:
            if model.__class__.__name__ == "Model_PI":
                X, y = X.float().to(device), y.to(device)
            else:
                X, y = [x.to(device) for x in X], y.to(device)
                y = y.argmax(1)
            pred = model(X)
            #return pred, y
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            preds.append(pred.argmax(1))
            ground_truth.append(y)
    test_loss /= num_batches
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct, preds, ground_truth


class Model_PI(nn.Module):
  def __init__(self,input_shape):
    super(Model_PI,self).__init__()
    self.fc1 = nn.Linear(input_shape,30)
    self.fc2 = nn.Linear(30,10)
    self.fc3 = nn.Linear(10,5)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x