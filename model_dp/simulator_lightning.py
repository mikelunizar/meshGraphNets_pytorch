import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from utils import normalization
import lightning.pytorch as pl
from .model import EncoderProcesserDecoder
from utils.utils import NodeTypeDP


class Simulator(pl.LightningModule):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, model_dir='checkpoint/simulator.pth', transforms=None):
        super().__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, node_input_size=node_input_size, edge_input_size=edge_input_size)
        self._output_normalizer = normalization.Normalizer(size=4, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer')
        self._edge_attr_normalizer = normalization.Normalizer(size=edge_input_size * 2, name='edge_normalizer')
        self._edge_world_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer')

        self.transforms = transforms

        print('Simulator model initialized')

    def forward(self, graph, mask):

        if self.training:

            node_type = graph.n
            x = graph.x
            target = graph.y
            velocity = target[:, :-1] - x[:, :-1]

            graph.x = self.__preprocess_node_attr(node_type, velocity)
            graph.edge_attr, graph.edge_world_attr = self.__preprocess_edge_attr(graph.edge_attr, graph.edge_world_attr)

            predicted_vel_stress = self.model(graph, mask)

            target_vel_stress = self.__position_stress_to_velocity_and_stress(x, target)
            target_vel_stress_normalized = self._output_normalizer(target_vel_stress[mask], self.training)

            return predicted_vel_stress, target_vel_stress_normalized
        
        else:
            node_type = graph.n
            x = graph.x
            target = graph.y
            velocity = target[:, :-1] - x[:, :-1]

            graph.x = self.__preprocess_node_attr(node_type, velocity)
            graph.edge_attr, graph.edge_world_attr = self.__preprocess_edge_attr(graph.edge_attr, graph.edge_world_attr)

            predicted_vel_stress = self.model(graph, mask)

            vel_stress_update = self._output_normalizer.inverse(predicted_vel_stress)
            predicted_position = x[:, :-1][mask] + vel_stress_update[:, :-1]
            predicted_stress = vel_stress_update[:, -2:-1]

            predicted_position_stress = torch.cat((predicted_position, predicted_stress), dim=-1)

            return predicted_position_stress

    def training_step(self, batch, batch_idx):

        graph = batch
        mask = torch.logical_or(batch.n.squeeze()==NodeTypeDP.NORMAL, batch.n.squeeze()==NodeTypeDP.WALL_BOUNDARY)

        predicted_vel_stress, target_vel_stress_normalized = self.forward(graph, mask)

        loss = F.mse_loss(predicted_vel_stress, target_vel_stress_normalized)
        
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=predicted_vel_stress.shape[0])
        
        return loss

    def validation_step(self, batch, batch_idx):

        graph = batch
        mask = torch.logical_or(batch.n.squeeze()==NodeTypeDP.NORMAL, batch.n.squeeze()==NodeTypeDP.WALL_BOUNDARY)
        target_position_stress = graph.y[mask]

        predicted_position_stress = self.forward(graph, mask)
        
        loss = F.mse_loss(predicted_position_stress, target_position_stress)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=predicted_position_stress.shape[0])
        
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        
        return optimizer
    
    def __get_noise(self, graph, noise_std=3e-3):

        pos = graph.pos
        type = graph.n.squeeze()
        noise = torch.normal(std=noise_std, mean=0.0, size=pos.shape).to(self.device)
        mask = type!=NodeTypeDP.NORMAL
        noise[mask]=0
        return noise.to(self.device)
        


    def __preprocess_node_attr(self, types, velocity):

        node_feature = []

        node_type = torch.squeeze(types.long())
        velocity[torch.argwhere(node_type != 1).squeeze()] = 0.
        node_feature.append(velocity)

        one_hot = F.one_hot(node_type, 4)
        node_feature.append(one_hot)

        node_feats = torch.cat(node_feature, dim=1).float()
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    def __preprocess_edge_attr(self, edge_attr, edge_world_attr):

        edge_attr = self._edge_attr_normalizer(edge_attr, self.training)
        edge_world_attr = self._edge_world_normalizer(edge_world_attr, self.training)

        return edge_attr, edge_world_attr

    def __position_stress_to_velocity_and_stress(self, x, target):

        velocity_next = target[:, :-1] - x[:, :-1]
        vel_stress_next = torch.cat((velocity_next, target[:, -2:-1]), dim=-1)

        return vel_stress_next


    def load_checkpoint(self, ckpdir=None):
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.' + k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s" % ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)

        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()
        _edge_attr_normalizer = self._edge_attr_normalizer.get_variable()
        _edge_world_normalizer = self._edge_world_normalizer.get_variable()

        to_save = {'model': model, '_output_normalizer': _output_normalizer, '_node_normalizer': _node_normalizer,
                   '_edge_attr_normalizer': _edge_attr_normalizer, '_edge_world_normalizer': _edge_world_normalizer}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s' % savedir)

    
