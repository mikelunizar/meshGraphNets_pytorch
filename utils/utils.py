import torch
from torch_geometric.data import Data
import enum

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9



# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_graph(graph):

    # graph: torch_geometric.data.data.Data
    return (graph.x, graph.edge_index, graph.edge_attr)

# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr = decompose_graph(graph)
    
    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    
    return ret

