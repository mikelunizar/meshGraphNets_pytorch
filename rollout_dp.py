from dataset import FPCdp_ROLLOUT
from torch_geometric.loader import DataLoader
import torch
import argparse
from tqdm import tqdm
import pickle
import torch_geometric.transforms as T
from utils.utils import NodeType, NodeTypeDP
import numpy as np
from model_dp.simulator import Simulator
from tqdm import tqdm
import os

from model_dp.transforms import FaceToEdgeTethra, RadiusGraphMesh, ContactDistance


parser = argparse.ArgumentParser(description='Implementation of MeshGraphNets')
parser.add_argument("--gpu",
                    type=int,
                    default=0,
                    help="gpu number: 0 or 1")

parser.add_argument("--model_dir",
                    type=str,
                    default='./checkpoint/simulator.pth')

parser.add_argument("--test_split", type=str, default='test')
parser.add_argument("--rollout_num", type=int, default=1)

args = parser.parse_args()

# gpu devices
torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout_error(predicteds, targets):

    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1), axis=0)/np.arange(1, number_len+1))

    for show_step in range(0, 1000000, 50):
        if show_step <number_len:
            print('testing rmse  @ step %d loss: %.2e'%(show_step, loss[show_step]))
        else: break

    return loss


@torch.no_grad()
def rollout(model, dataloader, rollout_index=1):

    dataset.change_file(rollout_index)

    predicted_position_stress = None
    mask=None
    predicteds = []
    targets = []

    for graph in tqdm(dataloader, total=400):

        if predicted_position_stress is not None:
            graph.x = predicted_position_stress.detach()

        graph = transformer(graph)
        graph = graph.cuda()

        if mask is None:
            node_type = graph.n
            mask = torch.logical_or(node_type==NodeTypeDP.NORMAL, node_type==NodeTypeDP.WALL_BOUNDARY)
            mask = torch.logical_not(mask).squeeze()
        
        predicted_position_stress = model(graph)

        next_position_stress = graph.y
        predicted_position_stress[mask] = next_position_stress[mask]

        predicteds.append(predicted_position_stress.detach().cpu().numpy())
        targets.append(next_position_stress.detach().cpu().numpy())
        
    result = [np.stack(predicteds), np.stack(targets)]

    os.makedirs('result', exist_ok=True)
    with open('result/result' + str(rollout_index) + '.pkl', 'wb') as f:
        pickle.dump([result, node_type], f)
    
    return result


if __name__ == '__main__':

    simulator = Simulator(message_passing_num=15, node_input_size=4, edge_input_size=4, device=device)
    simulator.load_checkpoint()
    simulator.eval()

    dataset_dir = "./data/deforming_plate"
    dataset = FPCdp_ROLLOUT(dataset_dir=dataset_dir, split='test')

    transformer = transformer = T.Compose([FaceToEdgeTethra(), T.Cartesian(norm=False), T.Distance(norm=False), 
                             RadiusGraphMesh(r=0.0003), ContactDistance(norm=False)])
    
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    for i in range(args.rollout_num):
        result = rollout(simulator, test_loader, rollout_index=i)
        print('------------------------------------------------------------------')
        rollout_error(result[0], result[1])


    



