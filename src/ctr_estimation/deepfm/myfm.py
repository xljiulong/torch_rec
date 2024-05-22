import numpy as np 
import pandas as pd
import random

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import os
import copy  

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device {}'.format(DEVICE))

class TorchFM(nn.Module):
    def __init__(self, n = None, k = None):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) # s_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # s^2

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out
    
train_df = pd.read_csv('/data/snlp/zhangjl/projects/torch_rec/data/dota_train_binary_heroes.csv', index_col='match_id_hash')
test_df = pd.read_csv('/data/snlp/zhangjl/projects/torch_rec/data/dota_test_binary_heroes.csv', index_col='match_id_hash')


target = pd.read_csv('/data/snlp/zhangjl/projects/torch_rec/data/train_targets.csv', index_col='match_id_hash')
y = target['radiant_win'].values.astype(np.float32)
y = y.reshape(-1, 1)

X_train = train_df.values.astype(np.float32)
X_test = test_df.values.astype(np.float32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def trainmlp(X, X_test, y, folds, model_class=None, model_params = None, batch_size=128, epochs=10, criterion=None,
             optimizer_class = None, opt_paras=None, device=None):
      seed_everything()
      models = []
      scores = []
      train_preds = np.zeros(y.shape)
      test_preds = np.zeros((X_test.shape[0], 1))

      X_tensor, X_test, y_tersor = torch.from_numpy(X).to(device), torch.from_numpy(X_test).to(device), torch.from_numpy(y).to(device)

      for n_fold, (train_ind, valid_ind) in enumerate(folds.split(X, y)):
          print(f'fold {n_fold + 1}')

          train_set = TensorDataset(X_tensor[train_ind], y_tersor[train_ind])
          valid_set = TensorDataset(X_tensor[valid_ind], y_tersor[valid_ind])

          loaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
                     'valid': DataLoader(valid_set, batch_size=batch_size, shuffle=True)}
          
          model = model_class(**model_params)
          model.to(device)
          best_model_wts = copy.deepcopy(model.state_dict())

          optimizer = optimizer_class(model.parameters(), **opt_paras)

          best_score = 0