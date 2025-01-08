import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing  import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pickle


def rot_dg(dg, device):
  dg_new = torch.zeros((dg.shape), device=device)

  dg_new[:, 0] = dg[:, 0]
  dg_new[:, 1] = dg[:, 1] - dg[:, 0]

  return dg_new


def gauss_filt(dg, centers, sigma, weight):
  # dg is shape (# points, 2)
  # centers is shape (# centers, 1, 2)

  ret = torch.pow(dg - centers, 2)
  ret = -torch.sum(ret, axis=2) / (2*torch.pow(sigma,2))
  #ret = torch.exp(ret)/(2*torch.pi*torch.pow(sigma, 2))
  ret = torch.exp(ret)/(2*torch.pi*torch.sqrt(sigma))

  if weight:
    ret = ret*dg[:,1]
  ret = torch.sum(ret, axis=1)

  return ret


def get_centers(x_lim, y_lim, x_n, y_n):
    x_steps = np.linspace(x_lim[0], x_lim[1], x_n)
    y_steps = np.linspace(y_lim[0], y_lim[1], y_n)

    cents = []
    for i in range(x_n-1, -1, -1):
        for j in range(y_n):
            cents.append([x_steps[j], y_steps[i]])

    centers = torch.tensor(cents)

    return centers.unsqueeze(1)


def preprocess_x(X, device, centers, sigma, mean, var, weighted, normalize=True):
    H0 = gauss_filt(rot_dg(X[0].T, device), centers, sigma, weight=weighted)
    H1 = gauss_filt(rot_dg(X[1].T, device), centers, sigma, weight=weighted)

    con = torch.concat((H0, H1))

    if normalize:
        variance = torch.from_numpy(var).to(device)
        cond = (con!=0) & (variance!=0)
        con[cond] = con[cond] - torch.from_numpy(mean).to(device)[cond]
        con[cond] = con[cond] / torch.sqrt(variance[cond])

    return con