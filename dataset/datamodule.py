import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl


import numpy as np
import os
import h5pickle as h5py
import os.path as osp


class DatasetDP(Dataset):

    def __init__(self, dataset_dir, split='test', trajectory=None):

        dataset_dir = osp.join(dataset_dir, split+'.h5')
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        self.data_keys = ("mesh_pos", "world_pos", "stress", "node_type", "cells")
        self.steps_trajectory = 400
        self.load_dataset(trajectory)

    @staticmethod
    def datas_to_graph(datas, num, step, metadata):

        face = torch.as_tensor(datas[metadata.index('cells')].T, dtype=torch.long)
        node_type = torch.as_tensor(datas[metadata.index('node_type')], dtype=torch.long)
        mesh_crds = torch.as_tensor(datas[metadata.index('mesh_pos')], dtype=torch.float)
        world_crds = torch.as_tensor(datas[metadata.index('world_pos')], dtype=torch.float)
        stress = torch.as_tensor(datas[metadata.index('stress')], dtype=torch.float)

        x = torch.cat((world_crds[0], stress[0]), dim=-1)
        y = torch.cat((world_crds[1], stress[1]), dim=-1)

        g = Data(x=x, y=y, face=face, n=node_type, pos=world_crds[0], mesh_pos=mesh_crds, num=num, step=step)
        
        return g

    def load_dataset(self, trajectory=None):

        self.dataset = []

        keys = list(self.file_handle.keys())
        self.trajectories = {k: self.file_handle[k] for k in keys}
        if trajectory is not None:
            self.trajectories = {str(trajectory): self.trajectories[str(trajectory)]}

        for num, trajectory in self.trajectories.items():

            for step in range(self.steps_trajectory-1):

                datas = []

                for k in self.data_keys:
                    if k in ["world_pos", 'stress']:
                        r = np.array((trajectory[k][step], trajectory[k][step+1]), dtype=np.float32)
                    else:
                        r = trajectory[k][step]
                        if k in ["node_type", "cells"]:
                            r = r.astype(np.int32)
                    datas.append(r)

                graph = DatasetDP.datas_to_graph(datas, num, step, self.data_keys)
                self.dataset.append(graph)

        print('Dataset loaded!')  
        print(f'    Number trajectories: {num}')
        print(f'    steps trajectory: {self.steps_trajectory}')
    
    def get_dataset(self):
        return self.dataset
    
    def get_loader(self, batch_size=32, shuffle=False, num_workers=1):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __len__(self):
        return len(self.datasets)
    


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers, transforms=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = 'train'
        self.valid_split = 'valid'
        self.transforms = transforms

    def setup(self, stage=None):
        self.train_dataset = DatasetDP(dataset_dir=self.dataset_dir, split=self.train_split)
        self.valid_dataset = DatasetDP(dataset_dir=self.dataset_dir, split=self.valid_split)

        if self.transforms is not None:
            print('Transforming data...')
            self.train_dataset.dataset = self.transforms(self.train_dataset.dataset)
            self.valid_dataset.dataset = self.transforms(self.valid_dataset.dataset)

    def train_dataloader(self, **kwargs):
        return DataLoader(self.train_dataset.dataset, num_workers=self.num_workers, pin_memory=True, **kwargs) 
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset.dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=True)
