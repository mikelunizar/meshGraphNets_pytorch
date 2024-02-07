import lightning.pytorch as pl
from utils.utils import NodeTypeDP
from torch_geometric.loader import DataLoader

import wandb
import torch
import numpy as np
import cv2
from pathlib import Path

from tqdm import tqdm

from render_results_dp import plot3D_position_stress


class RolloutCallback(pl.Callback):
    def __init__(self, rollout_loader, transforms):
        super().__init__()

        self.loader = rollout_loader
        self.transforms = transforms

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module):

        path = Path(trainer.checkpoint_callback.dirpath) / 'videos'
        path.mkdir(exist_ok=True, parents=True)

        results, n = self.rollout(pl_module, self.loader)
        
        path_rollout = self.make_video_from_results(results, n, trainer.current_epoch, path)

        trainer.logger.experiment.log({f"rollout": wandb.Video(path_rollout, format='mp4')})
        
    @torch.no_grad()
    def rollout(self, model, loader):
        mask = None            
        predicted_position_stress = None

        predicteds = []
        targets = []

        for graph in loader:

            if predicted_position_stress is not None:
                graph.x = predicted_position_stress.detach()

            graph = self.transforms(graph)
            graph = graph.to(model.device)

            if mask is None:
                node_type = graph.n
                mask = torch.logical_or(node_type==NodeTypeDP.NORMAL, node_type==NodeTypeDP.WALL_BOUNDARY)
                mask = torch.logical_not(mask).squeeze().cpu().detach().tolist()
            
            predicted_position_stress = model.forward(graph)

            next_position_stress = graph.y
            predicted_position_stress[mask] = next_position_stress[mask]

            predicteds.append(predicted_position_stress.detach().cpu().numpy())
            targets.append(next_position_stress.detach().cpu().numpy())
            
        result = [np.stack(predicteds), np.stack(targets)]

        return result, graph.n.squeeze().cpu().detach().numpy()
    
    def make_video_from_results(self, results, n, epoch, path):

        file_name = str(path) + '/rollout%d.mp4'%epoch

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_name, fourcc, 5, (1000, 800))

        skip=5
        
        def render(i):

            step = i * skip
            predicted = results[0][step]

            fig = plot3D_position_stress(predicted, n, step)

            fig.write_image(f'./frame.png', width=1000, height=800, scale=1)

            img = cv2.imread(f'./frame.png')
            img = cv2.resize(img, (1000, 800))
            out.write(img)

        for i in tqdm(range(399), total=400//skip):
            if i*skip < 350:
                render(i)
        out.release()
        print('video %s saved'%file_name)

        return file_name


    

        