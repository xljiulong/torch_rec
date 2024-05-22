from typing import Callable, List, Optional, Tuple, Union
import torch
from torch_geometric.data import InMemoryDataset, download_url, Dataset

class MyOwnInMemDataSet(InMemoryDataset):
    def __init__(self, root = None, transform = None, pre_transform = None, pre_filter = None, log = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return ['some_file1', 'some_file2']
        
        @property
        def processed_file_names(self):
            return ['data.pt']
        
        def download(self):
            data_list = [...]

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.prefilter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])



class MyOwnLargeDataSet(Dataset):
    def __init__(self, root: str = None, transform = None, pre_transform = None, pre_filter = None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file1', 'some_file2']
    
    @property
    def processed_file_names(self) -> str:
        return ['data_1.pt', 'data_2.pt']
    
    def download(self, url):
        path = download_url(url, self.raw_dir)

    def process(self):
        data = Dataset()
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = ''