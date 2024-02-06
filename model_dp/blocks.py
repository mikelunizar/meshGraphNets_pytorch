import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from utils.utils import decompose_graph
from torch_geometric.data import Data
import numpy as np


class EdgeMeshBlock(nn.Module):

    def __init__(self, custom_func=None):
        
        super(EdgeMeshBlock, self).__init__()
        self.net = custom_func


    def forward(self, graph):

        node_attr, edge_index, edge_attr, _, _ = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)
        
        edge_attr_ = self.net(collected_edges)   # Update

        return Data(x=node_attr, edge_attr=edge_attr_, edge_index=edge_index, edge_world_index=graph.edge_world_index, edge_world_attr=graph.edge_world_attr)
    

class EdgeWorldBlock(nn.Module):

    def __init__(self, custom_func=None):
        
        super(EdgeWorldBlock, self).__init__()
        self.net = custom_func


    def forward(self, graph):

        node_attr, _, _, edge_world_index, edge_world_attr = decompose_graph(graph)
        senders_idx, receivers_idx = edge_world_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_world_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)
        
        edge_world_attr_ = self.net(collected_edges)   # Update

        return Data(x=node_attr, edge_attr=graph.edge_attr, edge_index=graph.edge_index, edge_world_index=edge_world_index, edge_world_attr=edge_world_attr_)



    
class NodeBlock(nn.Module):

    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        node_attr, edge_index, edge_attr, edge_world_index, edge_world_attr = decompose_graph(graph)
        nodes_to_collect = []
        # Mesh
        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges_mesh = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)
        # Collision
        _, receivers_idx = graph.edge_world_index
        num_nodes = graph.num_nodes
        agg_received_edges_collision = scatter_add(edge_world_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(node_attr)
        nodes_to_collect.append(agg_received_edges_mesh)
        nodes_to_collect.append(agg_received_edges_collision)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        x = self.net(collected_nodes)

        return Data(x=x, edge_attr=edge_attr, edge_index=edge_index, edge_world_index=edge_world_index, edge_world_attr=edge_world_attr)
       
            
            
        