from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os



class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, device, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size =  node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, node_input_size=node_input_size, edge_input_size=edge_input_size).to(device)
        self._output_normalizer = normalization.Normalizer(size=4, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)
        self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        print('Simulator model initialized')

    def update_node_attr(self, types:torch.Tensor, velocity:torch.Tensor):
        node_feature = []

        node_type = torch.squeeze(types.long())
        velocity[torch.argwhere(node_type != 1).squeeze()] = 0.
        node_feature.append(velocity)

        one_hot = torch.nn.functional.one_hot(node_type, self.node_input_size)
        node_feature.append(one_hot)

        node_feats = torch.cat(node_feature, dim=1).float()
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    def position_to_velocity_and_stress(self, x, target):

        velocity_next = target[:,:-1] - x[:,:-1]
        vel_stress_next = torch.cat((velocity_next, target[:,-2:-1]), dim=-1)
        return vel_stress_next


    def forward(self, graph:Data):
        
        if self.training:
            
            node_type = graph.n
            x = graph.x
            target = graph.y
            velocity = target[:, :-1] - x[:, :-1]
            
            node_attr = self.update_node_attr(node_type, velocity) # noised_frames = frames + velocity_sequence_noise
            graph.x = node_attr

            predicted_vel_stress = self.model(graph)

            target_vel_stress = self.position_to_velocity_and_stress(x, target)
            target_vel_stress_normalized = self._output_normalizer(target_vel_stress, self.training)

            return predicted_vel_stress, target_vel_stress_normalized

        else:

            node_type = graph.n
            x = graph.x
            target = graph.y
            
            node_attr = self.update_node_attr(node_type) # noised_frames = frames + velocity_sequence_noise
            graph.x = node_attr

            predicted_vel_stress = self.model(graph)

            vel_stress_update = self._output_normalizer.inverse(predicted_vel_stress)
            predicted_position = x[:, :-1] + vel_stress_update[:, :-1]
            predicted_stress = vel_stress_update[:, -2:-1]

            predicted_position_stress = torch.cat((predicted_position, predicted_stress), dim=-1)

            return predicted_position_stress

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
                object = eval('self.'+k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s"%ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir=self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer  = self._node_normalizer.get_variable()
        _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {'model':model, '_output_normalizer':_output_normalizer, '_node_normalizer':_node_normalizer}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s'%savedir)