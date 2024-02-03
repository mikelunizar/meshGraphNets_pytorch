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
simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
optimizer= torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def plot_graph(graph):
    import matplotlib.pyplot as plt
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    positions = graph.x[:, 1:4]
    edges = graph.edge_index
    # Plot nodes
    for i, pos in enumerate(positions):
        color = graph.x[i, 0]
        ax.scatter(pos[0], pos[1], pos[2], c=color)

    # Plot edges
    for edge in edges.transpose(1, 0):
        x = [positions[edge[0]][0], positions[edge[1]][0]]
        y = [positions[edge[0]][1], positions[edge[1]][1]]
        z = [positions[edge[0]][2], positions[edge[1]][2]]
        ax.plot(x, y, z, color='gray', linestyle='dashed', linewidth=2)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def visualize_data(dataloader):
    for batch_index, graph in enumerate(dataloader):
        graph = FaceToEdgeTethra().forward(graph)
        plot_graph(graph)
        break


def train(model: Simulator, dataloader, optimizer):

    for batch_index, graph in enumerate(dataloader):

        graph = FaceToEdgeTethra().forward(graph)
        graph = transformer(graph)
        #graph = graph.cuda()
        plot_graph(graph)

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


#@functional_transform('face_to_edge_tethra')
class FaceToEdgeTethra():
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


from tqdm import tqdm


if __name__ == '__main__':

    #dataset_fpc = FPCdp(dataset_dir=dataset_dir, split='train', max_epochs=50)
    #train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=1)
    #transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])
    #visualize_data(train_loader)
    #train(simulator, train_loader, optimizer)
    dataset_fpc = FPCdp_ROLLOUT(dataset_dir=dataset_dir, split='test')
    rollout_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=1)
    transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])

    trajectory = []
    for i in range(1):
        dataset_fpc.change_file(i)
        for graph in tqdm(rollout_loader, total=400):
            graph = FaceToEdgeTethra().forward(graph)
            graph = transformer(graph)
            trajectory.append(graph)

    plot_simulation(trajectory)
