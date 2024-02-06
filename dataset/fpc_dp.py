from torch.utils.data import IterableDataset
import os, numpy as np
import os.path as osp
import h5pickle as h5py
from torch_geometric.data import Data
import torch
import math
import time


class FPCBase():

    def __init__(self, max_epochs=1, files=None):

        self.open_tra_num = 10
        self.file_handle = files
        self.shuffle_file() # read dataset into h5py files

        self.data_keys = ("mesh_pos", "world_pos", "stress", "node_type", "cells")
        self.out_keys = list(self.data_keys) + ['time']

        self.tra_index = 0
        self.epcho_num=1
        self.tra_readed_index = -1

        # dataset attr
        self.tra_len = 400
        self.time_iterval = 1

        self.opened_tra = []
        self.opened_tra_readed_index = {}
        self.opened_tra_readed_random_index = {}
        self.tra_data = {}
        self.max_epochs = max_epochs

    def open_tra(self):
        # Continue opening trajectories until the desired number is reached
        while len(self.opened_tra) < self.open_tra_num:

            # Get the index of the current trajectory
            tra_index = self.datasets[self.tra_index]

            # Check if the current trajectory is not already opened
            if tra_index not in self.opened_tra:
                # Add the trajectory index to the list of opened trajectories
                self.opened_tra.append(tra_index)

                # Initialize the read index for the current trajectory
                self.opened_tra_readed_index[tra_index] = -1

                # Create a random permutation of indices for the current trajectory
                self.opened_tra_readed_random_index[tra_index] = np.random.permutation(self.tra_len - 2)

            # Move to the next trajectory index
            self.tra_index += 1

            # Check if an epoch (cycle through the entire dataset) is completed
            if self.check_if_epcho_end():
                # Perform actions at the end of an epoch
                self.epcho_end()

                # Print a message indicating that the epoch has finished
                print('Epoch Finished')

    def check_and_close_tra(self):
        to_del = []
        for tra in self.opened_tra:
            if self.opened_tra_readed_index[tra] >= (self.tra_len - 3):
                to_del.append(tra)
        for tra in to_del:
            self.opened_tra.remove(tra)
            try:
                del self.opened_tra_readed_index[tra]
                del self.opened_tra_readed_random_index[tra]
                del self.tra_data[tra]
            except Exception as e:
                print(e)

    def shuffle_file(self):
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    def epcho_end(self):
        self.tra_index = 0
        self.shuffle_file()
        self.epcho_num = self.epcho_num + 1

    def check_if_epcho_end(self):
        if self.tra_index >= len(self.file_handle):
            return True
        return False

    @staticmethod
    def datas_to_graph(datas, metadata):

        face = torch.as_tensor(datas[metadata.index('cells')].T, dtype=torch.long)
        node_type = torch.as_tensor(datas[metadata.index('node_type')], dtype=torch.long)
        mesh_crds = torch.as_tensor(datas[metadata.index('mesh_pos')], dtype=torch.float)
        world_crds = torch.as_tensor(datas[metadata.index('world_pos')], dtype=torch.float)
        stress = torch.as_tensor(datas[metadata.index('stress')], dtype=torch.float)

        x = torch.cat((world_crds[0], stress[0]), dim=-1)
        y = torch.cat((world_crds[1], stress[1]), dim=-1)

        g = Data(x=x, y=y, face=face, n=node_type, pos=world_crds[0], mesh_pos=mesh_crds)
        
        return g

    def __next__(self):
   
        self.check_and_close_tra()
        self.open_tra()
        
        if self.epcho_num >self.max_epochs:
            raise StopIteration

        selected_tra = np.random.choice(self.opened_tra)

        data = self.tra_data.get(selected_tra, None)
        if data is None:
            data = self.file_handle[selected_tra]
            self.tra_data[selected_tra] = data

        selected_tra_readed_index = self.opened_tra_readed_index[selected_tra]
        selected_frame = self.opened_tra_readed_random_index[selected_tra][selected_tra_readed_index+1]
        self.opened_tra_readed_index[selected_tra] += 1

        datas = []
        for k in self.data_keys:
            if k in ["world_pos", 'stress']:
                r = np.array((data[k][selected_frame], data[k][selected_frame+1]), dtype=np.float32)
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)

        g = FPCBase.datas_to_graph(datas, self.data_keys)
  
        return g

    def __iter__(self):
        return self


class FPCdp(IterableDataset):

    def __init__(self, max_epochs, dataset_dir, split='train') -> None:

        super().__init__()

        dataset_dir = osp.join(dataset_dir, split+'.h5')
        self.max_epochs= max_epochs
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        print('Dataset '+  self.dataset_dir + ' Initialized')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_handle)
        else:
            per_worker = int(math.ceil(len(self.file_handle)/float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_handle))

        keys = list(self.file_handle.keys())
        keys = keys[iter_start:iter_end]
        files = {k: self.file_handle[k] for k in keys}
        return FPCBase(max_epochs=self.max_epochs, files=files)


class FPCdp_ROLLOUT(IterableDataset):
    def __init__(self, dataset_dir, split='test', name='flow pass a cylinder'):

        dataset_dir = osp.join(dataset_dir, split+'.h5')
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        self.data_keys = ("mesh_pos", "world_pos", "stress", "node_type", "cells")
        self.time_iterval = 1
        self.load_dataset()

    def load_dataset(self):
        datasets = list(self.file_handle.keys())
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)

    def change_file(self, file_index):
        
        file_index = self.datasets[file_index]
        self.cur_tra = self.file_handle[file_index]
        self.cur_targecity_length = self.cur_tra['mesh_pos'].shape[0]
        self.cur_tragecity_index = 0
        self.edge_index = None

    def __next__(self):
        if self.cur_tragecity_index == (self.cur_targecity_length - 1):
            raise StopIteration

        data = self.cur_tra
        selected_frame = self.cur_tragecity_index

        datas = []
        for k in self.data_keys:
            if k in ["world_pos", 'stress']:
                r = np.array((data[k][selected_frame], data[k][selected_frame + 1]), dtype=np.float32)
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)

        self.cur_tragecity_index += 1
        g = FPCBase.datas_to_graph(datas, self.data_keys)
        return g

    def __iter__(self):
        return self

