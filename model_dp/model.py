import torch.nn as nn
from .blocks import EdgeMeshBlock, EdgeWorldBlock, NodeBlock
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), 
                           #nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                           #nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                           nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128):
        super(Encoder, self).__init__()

        self.emb_encoder = build_mlp(edge_input_size * 2, hidden_size, hidden_size)
        self.ewb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)

        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        node_attr, _, edge_mesh_attr, _, edge_world_attr = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_mesh_ = self.emb_encoder(edge_mesh_attr)
        edge_world_ = self.ewb_encoder(edge_world_attr)
        
        return Data(x=node_, edge_attr=edge_mesh_, edge_index=graph.edge_index, edge_world_index=graph.edge_world_index, edge_world_attr=edge_world_)



class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()


        eb_input_dim = 3 * hidden_size
        nb_input_dim = 3 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        ewb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        emb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        
        
        self.emb_module = EdgeMeshBlock(custom_func=emb_custom_func)
        self.ewb_module = EdgeWorldBlock(custom_func=ewb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)

        graph = self.emb_module(graph)
        graph = self.ewb_module(graph)
        graph = self.nb_module(graph)

        edge_attr = graph_last.edge_attr + graph.edge_attr
        edge_world_attr = graph_last.edge_world_attr + graph.edge_world_attr
        x = graph_last.x + graph.x

        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index, edge_world_index=graph.edge_world_index, edge_world_attr=edge_world_attr)



class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph, mask):
        return self.decode_module(graph.x[mask])


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size)
        
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=4)

    def forward(self, graph, mask):

        graph= self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        
        decoded = self.decoder(graph, mask)

        return decoded







