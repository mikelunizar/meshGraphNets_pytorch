import torch_geometric
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from typing import Optional, Tuple



@functional_transform('face_to_edge_thetra')
class FaceToEdgeTethra(BaseTransform):
    r"""
    Adapted!
    Converts mesh faces :obj:`[3, num_faces]` to edge indices
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

            face = data.face
            edge_index = torch.cat([face[0:2], face[1:3], face[2::], face[0::2], face[1::2], face[0::3]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.x.shape[0])

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data
    

@functional_transform('radius_graph_mesh')
class RadiusGraphMesh(BaseTransform):
    r"""
        Adapted!

        Creates edges based on node positions :obj:`data.pos` to all points
    within a given distance (functional name: :obj:`radius_graph`).

    Args:
        r (float): The distance.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            This flag is only needed for CUDA tensors. (default: :obj:`32`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
    """
    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        num_workers: int = 1,
    ):
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

    def forward(self, data: Data) -> Data:

        batch = data.batch if 'batch' in data else None

        data.edge_world_index = torch_geometric.nn.radius_graph(
            data.pos,
            self.r,
            batch,
            self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
            num_workers=self.num_workers,
        )


        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'
    

@functional_transform('contact_distance')
class ContactDistance(BaseTransform):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes
    (functional name: :obj:`distance`). Each distance gets globally normalized
    to a specified interval (:math:`[0, 1]` by default).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
        interval ((float, float), optional): A tuple specifying the lower and
            upper bound for normalization. (default: :obj:`(0.0, 1.0)`)
    """
    def __init__(
            self,
            norm: bool = True,
            max_value: Optional[float] = None,
            cat: bool = True,
            interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        (row, col), pos = data.edge_world_index, data.pos

        cart_dist = pos[col] - pos[row]
        dist_norm = torch.norm(cart_dist, p=2, dim=-1).view(-1, 1)
        dist = torch.cat((cart_dist, dist_norm), dim=-1)


        if self.norm and dist.numel() > 0:
            max_value = dist.max() if self.max is None else self.max

            length = self.interval[1] - self.interval[0]
            dist = length * (dist / max_value) + self.interval[0]

        data.edge_world_attr = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')