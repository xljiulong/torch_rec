import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2], 
                           [1, 0, 2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

Data(edge_index=[2, 4], x=[3, 1])



edge_index = torch.tensor([[0,1],
                           [1,0],
                           [1,2],
                           [2,1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x = x, edge_index=edge_index.t().contiguous())


for key, item in data:
    print(f'{key} found in data')

print(data.num_node_features)



print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='../data/enzymes', name='ENZYMES')
