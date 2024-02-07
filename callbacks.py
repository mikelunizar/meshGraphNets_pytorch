import lightning.pytorch as pl
from utils.utils import NodeTypeDP
from torch_geometric.loader import DataLoader

import wandb
import torch
from copy import deepcopy
import numpy as np
import pickle
import cv2

from render_results_dp import plot3D_position_stress


class RolloutCallback(pl.Callback):
    def __ini__(self, rollout_loader, trajectory, transforms, **kwargs):
        super().__init__(**kwargs)

        loader_dict = {sample.step: sample for sample in rollout_loader if sample.num == trajectory}
        sorted_dict = dict(sorted(loader_dict.items()))
        self.loader = DataLoader([graph for step, graph in sorted_dict.items()])
        
        self.transforms = transforms

    def rollout(self, model, loader):
        mask = None            
        predicted_position_stress = None

        predicteds = []
        targets = []

        for graph in loader:

            if predicted_position_stress is not None:
                graph.x = predicted_position_stress.detach()

            graph = self.transforms(graph)

            if mask is None:
                node_type = graph.n
                mask = torch.logical_or(node_type==NodeTypeDP.NORMAL, node_type==NodeTypeDP.WALL_BOUNDARY)
                mask = torch.logical_not(mask).squeeze()
            
            predicted_position_stress = model.forward(graph)

            next_position_stress = graph.y
            predicted_position_stress[mask] = next_position_stress[mask]

            predicteds.append(predicted_position_stress.detach().cpu().numpy())
            targets.append(next_position_stress.detach().cpu().numpy())
            
        result = [np.stack(predicteds), np.stack(targets)]

        return result
    
    def make_video_from_results(results, epoch):

        for index, file in enumerate(results):

            with open(file, 'rb') as f:
                result, n = pickle.load(f)
            n = n.cpu().detach().numpy()

            file_name = '.videos/output%d.mp4'%epoch

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_name, fourcc, 10.0, (1000, 800))

            skip=5
            
            def render(i):

                step = i * skip
                predicted = result[0][step]

                fig = plot3D_position_stress(predicted, n, step)

                fig.write_image(f'./frame.png', width=1000, height=800, scale=2)

                img = cv2.imread(f'./frame.png')
                img = cv2.resize(img, (1000, 800))
                out.write(img)


            for i in tqdm(range(399), total=400//skip):
                if i*skip < 350:
                    render(i)
            out.release()
            print('video %s saved'%file_name)

            return file_name


    def on_validation_end(self, trainer, pl_module):

        results = self.rollout(pl_module, self.loader)
        path_rollout = self.make_video_from_results(results, trainer.current_epoch)

        trainer.logger.experiment.log({f"rollout": wandb.Video(path_rollout, format='mp4')})
        

        