from dataset import FPCdp
from model_dp.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

dataset_dir = "./data/deforming_plate"
batch_size = 1
noise_std=2e-2

print_batch = 10
save_batch = 200


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
optimizer= torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def train(model:Simulator, dataloader, optimizer):

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        #node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
        #velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph)#, velocity_sequence_noise)
        mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
        
        errors = ((predicted_acc - target_acc)**2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print('batch %d [loss %.2e]'%(batch_index, loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint()



from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected


@functional_transform('face_to_edge_tethra')
class FaceToEdgeTethra(BaseTransform):
    r"""Converts mesh faces :obj:`[3, num_faces]` to edge indices
    :obj:`[2, num_edges]` (functional name: :obj:`face_to_edge`).

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
    """
    def __init__(self, remove_faces: bool = True):
        self.remove_faces = remove_faces

    def forward(self, data: Data) -> Data:
        if hasattr(data, 'face'):
            face = data.face
            num_edges_element = face.shape[0]
            list_elements = [i for i in range(num_edges_element)]
            # Using nested loops to generate all pairs
            all_combinations = [(x, y) for x in list_elements for y in list_elements if x != y]
            edge_index = []
            for (x, y) in all_combinations:
                single_combination_edge_index = torch.cat((face[x,:].unsqueeze(0), face[y, :].unsqueeze(0)), dim=0)
                edge_index.append(single_combination_edge_index)
            edge_index = torch.cat(edge_index, dim=-1)
            #edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data

if __name__ == '__main__':

    dataset_fpc = FPCdp(dataset_dir=dataset_dir, split='train', max_epochs=50)
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=1)
    transformer = T.Compose([FaceToEdgeTethra()])#, T.Cartesian(norm=False), T.Distance(norm=False)])
    train(simulator, train_loader, optimizer)
