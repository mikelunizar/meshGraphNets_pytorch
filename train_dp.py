from dataset import FPCdp, FPCdp_ROLLOUT
from model_dp.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
from model_dp.transforms import FaceToEdgeTethra, RadiusGraphMesh, ContactDistance
from utils.utils import NodeTypeDP
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


dataset_dir = "./data/deforming_plate"
batch_size = 1
noise_std=2e-2

print_batch = 10
save_batch = 200


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=7, edge_input_size=4, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def train(model: Simulator, dataloader, optimizer):

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.to(device)

        node_type = graph.n  # "node_type, mesh_pos, world_pos, stress"
        # ADD NOISE
        #velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph)
        # Mask data to not backpropagate no error actuator
        mask = torch.logical_or(node_type==NodeTypeDP.NORMAL, node_type==NodeTypeDP.WALL_BOUNDARY).squeeze()
        
        errors = ((predicted_acc - target_acc)**2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print('batch %d [loss %.2e]'%(batch_index, loss.item()))

        if batch_index % save_batch == 0:
            model.save_checkpoint()




if __name__ == '__main__':

    dataset_fpc = FPCdp(dataset_dir=dataset_dir, split='train', max_epochs=50)
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=1)
    transformer = T.Compose([FaceToEdgeTethra(), T.Cartesian(norm=False), T.Distance(norm=False), 
                             RadiusGraphMesh(r=0.0003), ContactDistance(norm=False)])
    train(simulator, train_loader, optimizer)


