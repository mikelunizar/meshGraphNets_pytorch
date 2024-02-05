from dataset import FPCdp, FPCdp_ROLLOUT
from model_dp.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

dataset_dir = "./data/deforming_plate"
batch_size = 1
noise_std=2e-2

print_batch = 10
save_batch = 200


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=4, edge_input_size=4, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def train(model: Simulator, dataloader, optimizer):

    for batch_index, graph in enumerate(dataloader):
        print(batch_index)

        graph = transformer(graph)
        graph = graph.to(device)

        node_type = graph.n  # "node_type, mesh_pos, world_pos, stress"
        # ADD NOISE
        #velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph)
        # Mask data to not backpropagate no error
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

@functional_transform('face_to_edge_thetra')
class FaceToEdgeTethra(BaseTransform):
    r"""Converts mesh faces :obj:`[3, num_faces]` to edge indices
    :obj:`[2, num_edges]` (functional name: :obj:`face_to_edge`).

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
    """
    def __init__(self, remove_faces: bool = True):
        super().__init__()
        self.remove_faces = remove_faces

    def forward(self, data: Data) -> Data:
        if hasattr(data, 'face'):
            # face = data.face
            # type_nodes = data.n.squeeze()
            # num_edges_element = face.shape[0]

            #edge_index = to_undirected(edge_index, num_nodes=data.x.shape[0])
        
            # idx_object = torch.argwhere(data.n != 1)[:, 0]
            # idx_actuator = torch.argwhere(data.n == 1)[:, 0]

            # list_actuator, list_object = [], []
            # for i in range(data.x.shape[0]):
            #     if i in idx_actuator:
            #         list_actuator.append(i)
            #     elif i in idx_object:
            #         list_object.append(i)

            # face_object_idx = torch.unique(torch.argwhere(face < list_actuator[0])[:, -1])
            # face_actuator_idx = torch.unique(torch.argwhere(face >= list_actuator[0])[:, -1])

            # face_object = face[:, face_object_idx]
            # face_actuator = face[:, face_actuator_idx]

            # list_elements = [i for i in range(num_edges_element)]
            # # Using nested loops to generate all pairs
            # all_combinations = [(x, y) for x in list_elements for y in list_elements if x != y]
            # edge_index = []
            # for (x, y) in all_combinations:
            #     single_combination_edge_index = torch.cat((face[x, :].unsqueeze(0).tolist(), face[y, :].unsqueeze(0).tolist()), dim=0)
            #     edge_index.append(single_combination_edge_index)

            # edge_index = torch.cat(edge_index, dim=-1)

            face = data.face
            edge_index = torch.cat([face[0:2], face[1:3], face[2::], face[0::2], face[1::2], face[0::3]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.x.shape[0])

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data



if __name__ == '__main__':

    dataset_fpc = FPCdp(dataset_dir=dataset_dir, split='train', max_epochs=50)
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=1)
    transformer = T.Compose([FaceToEdgeTethra(), T.Cartesian(norm=False), T.Distance(norm=False)]) # 
    train(simulator, train_loader, optimizer)



