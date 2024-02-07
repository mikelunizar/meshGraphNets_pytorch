import torch
import lightning.pytorch as pl
import datetime
from pathlib import Path
import os.path as osp

import torch_geometric.transforms as T
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model_dp.simulator_lightning import Simulator
from model_dp.callbacks import RolloutCallback
from model_dp.transforms import FaceToEdgeTethra, RadiusGraphMesh, ContactDistance, MeshDistance
from dataset.datamodule import DataModule, DatasetDP


pl.seed_everything(42, workers=True)

batch_size = 25
noise_std = 3e-3

name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
chckp_path = Path(f'outputs/runs/{name}')
chckp_path.mkdir(exist_ok=True, parents=True)

dataset_dir="./data/deforming_plate_debug"
wandb_logger = WandbLogger(name=name, project='MeshGraph')
checkpoint = ModelCheckpoint(dirpath=chckp_path._str, monitor='val_loss', save_top_k=3)


transforms = T.Compose([FaceToEdgeTethra(), T.Cartesian(norm=False), T.Distance(norm=False), 
                            RadiusGraphMesh(r=0.03), MeshDistance(norm=False), ContactDistance(norm=False)])

data_module = DataModule(dataset_dir=dataset_dir, 
                        batch_size=batch_size, 
                        num_workers=1, 
                        transforms=transforms)

rollout_data = DatasetDP(dataset_dir=dataset_dir, split='train', trajectory=0)
rollout_loader = rollout_data.get_loader(batch_size=1, shuffle=False, num_workers=1)

rollout = RolloutCallback(rollout_loader, transforms)


simulator = Simulator(message_passing_num=15, node_input_size=7, edge_input_size=4,
                        transforms=transforms)

trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else 0, 
                        max_epochs=100000, 
                        logger=wandb_logger,
                        callbacks=[checkpoint, rollout],
                        num_sanity_val_steps=0,
                        deterministic=True,
                        check_val_every_n_epoch=25
                        )
    
trainer.fit(simulator, datamodule=data_module)
