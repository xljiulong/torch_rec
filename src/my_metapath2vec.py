# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/metapath2vec.py
import os.path as osp
import os
import torch
from torch_geometric.datasets import AMiner
from torch_geometric.nn import MetaPath2Vec
import torch.multiprocessing as mp
from datetime import datetime


path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/AMiner')
path = '/data/snlp/zhangjl/projects/torch_rec/data/AMiner'

dataset = AMiner(path)
data = dataset[0]

metapath = [
    ('author', 'writes', 'paper'),
    ('paper', 'published_in', 'venue'),
    ('venue', 'publishes', 'paper'),
    ('paper', 'written_by', 'author'),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# torch.set_num_threads(42)
# os.environ["OMP_NUM_THREADS"] = "42"  # 设置OpenMP计算库的线程数
# os.environ["MKL_NUM_THREADS"] = "42"  # 设置MKL-DNN CPU加速库的线程数。

model = MetaPath2Vec(data.edge_index_dict, embedding_dim=64, metapath=metapath, walk_length=20, 
                     context_size=5, walks_per_node=5, num_negative_samples=5,
                     sparse=True).to(device)

loader = model.loader(batch_size = 1024, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=20)

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train(epoch, log_steps=50, eval_steps=200):
    print('begin training')
    model.train()
    total_loss = 0

    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()

        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {total_loss / log_steps:.4f}, '
                   f'Time: {datetime.now()}'))
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            acc = test()
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Acc: {acc:.4f}'))
            
@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model('author', batch=data['author'].y_index.to(device))
    y = data['author'].y

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                      max_iter=150)

for epoch in range(1, 6):
    train(epoch)
    acc = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')