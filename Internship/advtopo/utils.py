import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing  import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pickle

from advtopo import pis


def load_data(dataset, path_dataset="", filtrations=[], verbose=False):

    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    diagfile = h5py.File(path_dataset + dataset + ".hdf5", "r")
    filts = list(diagfile.keys()) if len(filtrations) == 0 else filtrations

    diags_dict = dict()
    if len(filts) == 0:
        filts = diagfile.keys()
    for filtration in filts:
        list_dgm, num_diag = [], len(diagfile[filtration].keys())
        for diag in range(num_diag):
            list_dgm.append(np.array(diagfile[filtration][str(diag)]))
        diags_dict[filtration] = list_dgm

    # Extract features and encode labels with integers
    feat = pd.read_csv(path_dataset + dataset + ".csv", index_col=0, header=0)
    F = np.array(feat)[:, 1:]  # 1: removes the labels
    L = np.array(LabelEncoder().fit_transform(np.array(feat["label"])))
    L = OneHotEncoder(sparse=False, categories="auto").fit_transform(L[:, np.newaxis])

    if verbose:
        print("Dataset:", dataset)
        print("Number of observations:", L.shape[0])
        print("Number of classes:", L.shape[1])

    return diags_dict, F, L


class OrbitsDataset(Dataset):
    def __init__(self, D, L):
        self.D = D
        self.L = L

    def __len__(self):
        return self.D[0].shape[0]

    def __getitem__(self, idx):
        h0_pds = torch.from_numpy(np.transpose(self.D[0][idx,:,:2]))
        h1_pds = torch.from_numpy(np.transpose(self.D[1][idx,:,:2]))
        samples = (h0_pds, h1_pds)
        labels = torch.from_numpy(self.L[idx]).to(torch.float32)
        
        return (samples, labels)
    

class PIDataset(Dataset):
  def __init__(self,x,y):
    #self.x = torch.tensor(x,dtype=torch.float32)
    #self.y = torch.tensor(y,dtype=torch.float32)
    self.x = x
    self.y = y
    self.length = len(x)
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length


def get_torch_datasets():

    dirPath = "./data/"

    # We only use L from here, the one-hot encoded labels
    diags_dict, F, L = load_data("ORBIT5K", path_dataset=dirPath)

    # D is preprocessed as in Perslay
    with open(dirPath + 'processed_D.pkl', 'rb') as f:
        D = pickle.load(f)

    D0_train, D0_test, D1_train, D1_test, L_train, L_test = train_test_split(D[0], D[1], L, test_size=0.3)
    D_train = [D0_train, D1_train]
    D_test = [D0_test, D1_test]

    ds_train = OrbitsDataset(D_train, L_train)
    ds_test = OrbitsDataset(D_test, L_test)

    return ds_train, ds_test


def convert_orbit_to_pi_dataset(orbit_ds, device, centers, sigma, mean, var, weighted):
    Xs = []
    ys = []

    for (X, y) in orbit_ds:

        H0 = pis.gauss_filt(pis.rot_dg(X[0].T, device), centers, sigma, weight=weighted)
        H1 = pis.gauss_filt(pis.rot_dg(X[1].T, device), centers, sigma, weight=weighted)

        con = torch.concat((H0, H1))
        con = con - torch.from_numpy(mean).to(device)
        con = con / torch.sqrt(torch.from_numpy(var)).to(device)
        con[((con==np.inf) | (con==-np.inf)  | (torch.isnan(con)))] = 0
        Xs.append(con)
        ys.append(torch.argmax(y).to(device))
        

    return Xs, ys