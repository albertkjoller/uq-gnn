import torch
import numpy as np
import networkx as nx
from numba import jit
from dgl.data import MiniGCDataset, QM9Dataset
from tqdm import tqdm

class GraphDataset():
    """Parent class for graph datasets.

    Provides some convenient properties and functions.

    To create a dataset, inherit from this class and specify the following
    member variables

    Member variables
    ----------------
    num_graphs        : Number of graphs
    node_coordinates  : 2-d coordinates of all nodes in all graphs
                        (shape num_nodes x 2)
    node_graph_index  : Graph index (between 0 and num_graphs-1) for each node
                        (shape num_nodes)
    edge_list         : Array of edges (shape num_edges x 2)
    """

    def __init__(self, dim=2):
        self.dim = dim
        self.num_graphs = 0
        self.node_coordinates = torch.empty((0, dim))
        self.node_graph_index = torch.empty((0))
        self.edge_list = torch.empty((0, 2))
        self._current_angle = 0.

    @property
    def node_from(self):
        """List of first nodes for each edge = edge_list[:, 0]."""
        return self.edge_list[:, 0]

    @property
    def node_to(self):
        """: List of first nodes for each edge = edge_list[:, 0]."""
        return self.edge_list[:, 1]

    @property
    def edge_graph_index(self):
        """Graph index for each edge."""
        return self.node_graph_index[self.node_to]

    @property
    def num_nodes(self):
        """Number of nodes."""
        return self.node_coordinates.shape[0]

    @property
    def num_edges(self):
        """Number of edges."""
        return self.edge_list.shape[0]

    @property
    def edge_vector_diffs(self):
        """Vector between nodes for each edge."""
        return (self.node_coordinates[self.node_to] -
                self.node_coordinates[self.node_from])

    @property
    def edge_lengths(self):
        """Length of each edge."""
        return self.edge_vector_diffs.norm(dim=1, keepdim=True)

    @property
    def edge_vectors(self):
        """Normalized vector between nodes for each edge."""
        return self.edge_vector_diffs / self.edge_lengths

    def center(self):
        """Centers each graph in the dataset on its mean coordinate.

        Returns
        -------
        GraphDataset
            Returns self to enable chaining operations

        """
        for i in range(self.num_graphs):
            coords = self.node_coordinates[self.node_graph_index == i]
            mean = coords.mean(0)
            self.node_coordinates[self.node_graph_index == i] = coords-mean
        return self

    def rotate(self, angle, axis=None):
        """Rotates each graph in the data set individually around it's center.

        Parameters
        ----------
        angle : float
            Rotation angle (in radians)

        Returns
        -------
        GraphDataset
            Returns self to enable chaining operations

        """
        relative_angle = angle - self._current_angle
        self._current_angle = angle
        phi = torch.tensor(relative_angle)
        
        if self.dim == 2:
            R = torch.tensor([[torch.cos(phi), -torch.sin(phi)],
                             [torch.sin(phi), torch.cos(phi)]])
        
        if self.dim == 3:
            assert axis != None, "Specify dimension on which to rotate (either 0, 1 or 2)"
            
            # Setup rotation matrix
            R = torch.zeros(3, 3)
            r_ax1, r_ax2 = torch.arange(3)[torch.arange(3) != axis]

            # Create rotation matrix
            R[axis, axis] = 1
            R[r_ax1, r_ax1] = torch.cos(phi)
            R[r_ax1, r_ax2] = -torch.sin(phi)
            R[r_ax2, r_ax1] = torch.sin(phi)
            R[r_ax2, r_ax2] = torch.cos(phi)  
            
        for i in range(self.num_graphs):
            coords = self.node_coordinates[self.node_graph_index == i]
            mean = coords.mean(0)
            self.node_coordinates[self.node_graph_index == i] = (
                coords-mean) @ R + mean
        return self
    
    
class SyntheticData(GraphDataset):
    """Synthetic Data á la Felix!
    

    Loads a data set that was created in XXXXX.py
    """
    def __init__(self, data, graph_idx, graph_coords, test_size=0.1, device='cpu', seed = 42):
        #super().__init__(dim=)

        # Store input arguments in class
        self.device = device
        self.num_graphs = int(graph_idx.max())
        self.graph_coords = graph_coords
        torch.manual_seed(seed)


        # Split train and test data
        idxs = np.arange(self.num_graphs)
        train_idxs = np.random.choice(idxs, int(self.num_graphs * (1 - test_size)), replace=False)
        test_idxs = np.setdiff1d(idxs, train_idxs)
        train_graphs = self.get_info(data, train_idxs, graph_idx, graph_coords)
        test_graphs = self.get_info(data, train_idxs, graph_idx, graph_coords)

        # Store information
        self.data = dict({'train': GraphDataset(), 'test': GraphDataset()})
        for dtype, graphs_ in {'train': train_graphs, 'test': test_graphs}.items():
            self.data[dtype].node_coordinates = graphs_[0].to(self.device)
            self.data[dtype].node_graph_index = graphs_[1].to(self.device)
            self.data[dtype].edge_list = graphs_[2].to(self.device)
            self.data[dtype].target = graphs_[3].unsqueeze(1).to(self.device)
            self.data[dtype].num_graphs = graphs_[3].__len__()
        x=1
    def get_info(self,data, idxs, graph_idx, graph_coords):
        new_coords = torch.tensor([])
        new_graph_idx = torch.tensor([])
        new_edge_list = torch.tensor([])
        target = torch.tensor([])
        for count, i in enumerate(idxs):
            if count==0:
                nc = graph_coords[torch.unique(data[torch.where(data[:,-1]==graph_idx[i])[0],0]).to(torch.long)]
                new_coords = nc
                new_graph_idx = graph_idx[i]*torch.ones(nc.__len__())
                new_edge_list = data[torch.where(data[:, -1] == graph_idx[i])[0]][:, :2]
                target = data[i, 3]
            else:
                nc = graph_coords[torch.unique(data[torch.where(data[:,-1]==graph_idx[i])[0],0]).to(torch.long)]
                new_coords = torch.vstack((new_coords, nc))
                new_graph_idx = torch.hstack((new_graph_idx, graph_idx[i]*torch.ones(nc.__len__())))
                new_edge_list = torch.vstack((new_edge_list, data[torch.where(data[:,-1]==graph_idx[i])[0]][:,:2]))
                target = torch.hstack((target, data[i,3]))
        return new_coords.to(torch.float), new_graph_idx.to(torch.long), new_edge_list.to(torch.long), target.to(torch.float)
    #@jit(forceobj=True)
    def get_information(self, data, idxs):
        res = np.empty(len(idxs), dtype=np.float64)

        node_coordinates = torch.tensor([])
        node_graph_index = torch.tensor([])
        edge_list = torch.tensor([])

        pgl = 0
        for i, idx in enumerate(idxs):
            g, label = data[idx]

            res[i] = label.item()
            coords_ = g.ndata['R']
            node_coordinates = torch.concat([node_coordinates, coords_])
            node_graph_index = torch.concat([node_graph_index, i * torch.ones([coords_.__len__()])])

            node_from, node_to = g.edges(form='uv')
            edge_list = torch.concat([edge_list, pgl + torch.stack(g.edges()).T])

            # Previous graph length
            pgl += coords_.__len__()

        # Target attributes
        #graph_list = torch.tensor(res > res.mean())
        return node_coordinates, node_graph_index.to(torch.long), edge_list.to(torch.long), torch.tensor(res).to(torch.float)


class MiniGCData(GraphDataset):
    """MiniGCDataset dataset.

    Creates a variety of synthetic graphs.
    """

    def __init__(self, num_graphs=100, min_num_v=4, max_num_v=10, seed=42):
        super().__init__(dim=2)
        
        # Class index --> class name
        self.numClasses = 8
        self.classNames = {0: 'cycle graph', 1: 'star graph', 2: 'wheel graph',
                           3: 'lollipop graph', 4: 'hypercube graph', 5: 'grid graph',
                           6: 'clique graph', 7: 'circular ladder graph'}
        
        # Set seed
        self.seed = seed
        
        # Number of graphs in the dataset
        self.num_graphs = num_graphs
        
        # Generate dataset
        data_ = MiniGCDataset(self.num_graphs, min_num_v=min_num_v, max_num_v=max_num_v, seed=self.seed)
        
        # Initialize graph attributes
        self.node_coordinates = torch.tensor([])
        self.node_graph_index = torch.tensor([])
        
        # Initialize label attributes
        self.graph_labels = {}
        self.graph_list = torch.empty(num_graphs, ).to(torch.long)
        
        pgl = 0
        
        # Restructure DGL-style graphs
        for index, (g, label) in enumerate(data_):
            # Construct networkx object
            G_ = nx.DiGraph(g.to_networkx())
            G_.remove_edges_from(nx.selfloop_edges(G_))
            
            # Sort nodes
            G = nx.DiGraph()
            G.add_nodes_from(sorted(G_.nodes(data=True)))
            G.add_edges_from(G_.edges(data=True))
            del G_
            
            # Node coordinates
            G_coords = torch.tensor([list(pos) for pos in nx.spring_layout(G, seed=self.seed).values()])
            self.node_coordinates = torch.concat([
                self.node_coordinates, G_coords,
            ]).to(torch.float)
            
            # Node graph index
            self.node_graph_index = torch.concat([self.node_graph_index, 
                                                  index * torch.ones(G.__len__())]).to(torch.long)
            
            # Edge list
            self.edge_list = torch.concat([self.edge_list,
                                           pgl + torch.tensor([list(edge) for edge in G.edges])
                                         ]).to(torch.long)
                    
            # Graph labels
            self.graph_labels[index] = self.classNames[label.item()]
            self.graph_list[index] = label.item()
            
            # Previous graph length
            pgl += G.__len__() 


class QM9:
    """Molecular dataset.

    
    """

    def __init__(self, label_attr: str, num_graphs=None, test_size=0.1, device='cpu'):
        #super().__init__(dim=)

        # Store input arguments in class
        self.num_graphs = num_graphs
        self.device = device
        
        # Load graph data
        data = QM9Dataset(label_keys=[label_attr])
        num_graphs = num_graphs if num_graphs != None else data.__len__()

        # Split train and test data
        idxs = np.arange(num_graphs)
        train_idxs = np.random.choice(idxs, int(num_graphs * (1 - test_size)), replace=False)
        test_idxs = np.setdiff1d(idxs, train_idxs)
        train_graphs = self.get_information(data, train_idxs)
        test_graphs = self.get_information(data, test_idxs)

        # Store information
        self.data = dict({'train': GraphDataset(), 'test': GraphDataset()})
        for dtype, graphs_ in {'train': train_graphs, 'test': test_graphs}.items():
            self.data[dtype].node_coordinates = graphs_[0].to(self.device)
            self.data[dtype].node_graph_index = graphs_[1].to(self.device)
            self.data[dtype].edge_list = graphs_[2].to(self.device)
            self.data[dtype].target = graphs_[3].unsqueeze(1).to(self.device)
            self.data[dtype].num_graphs = graphs_[3].__len__()
    
    @jit(forceobj=True)
    def get_information(self, data, idxs):
        res = np.empty(len(idxs), dtype=np.float64)

        node_coordinates = torch.tensor([])
        node_graph_index = torch.tensor([])
        edge_list = torch.tensor([])

        pgl = 0
        for i, idx in enumerate(idxs):
            g, label = data[idx]

            res[i] = label.item()
            coords_ = g.ndata['R']
            node_coordinates = torch.concat([node_coordinates, coords_])
            node_graph_index = torch.concat([node_graph_index, i * torch.ones([coords_.__len__()])])

            node_from, node_to = g.edges(form='uv')
            edge_list = torch.concat([edge_list, pgl + torch.stack(g.edges()).T])
            
            # Previous graph length
            pgl += coords_.__len__() 
        
        # Target attributes
        #graph_list = torch.tensor(res > res.mean())
        return node_coordinates, node_graph_index.to(torch.long), edge_list.to(torch.long), torch.tensor(res).to(torch.float)