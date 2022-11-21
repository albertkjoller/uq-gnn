import torch
import numpy as np
import networkx as nx
from numba import jit
from dgl.data import MiniGCDataset, QM9Dataset
import scipy.io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split, BatchSampler, SequentialSampler
from itertools import combinations



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
        self.num_graphs = int(graph_idx.max())+1
        self.graph_coords = graph_coords
        torch.manual_seed(seed)


        # Split train and test data
        idxs = np.arange(self.num_graphs)


        graphs = self.get_info(data, idxs, graph_idx, graph_coords)


        # Store information
        self.data = GraphDataset()
        self.data.node_coordinates = graphs[0].to(self.device)
        self.data.node_graph_index = graphs[1].to(self.device)
        self.data.edge_list = graphs[2].to(self.device)
        self.data.target = graphs[3].unsqueeze(1).to(self.device)
        self.data.num_graphs = graphs[3].__len__()

    def get_info(self,data, idxs, graph_idx, graph_coords):
        new_coords = torch.tensor([])
        new_graph_idx = torch.tensor([])
        new_edge_list = torch.tensor([])
        target = torch.tensor([])
        for count, i in enumerate(idxs):
            if count==0:
                nc = graph_coords[torch.unique(data[torch.where(data[:,-1]==i)[0],0]).to(torch.long)]
                new_coords = nc
                new_graph_idx = graph_idx[i]*torch.ones(nc.__len__())
                new_edge_list = data[torch.where(data[:, -1] == i)[0]][:, :2]
                target = data[i, 3]
            else:
                nc = graph_coords[torch.unique(data[torch.where(data[:,-1]==i)[0],0]).to(torch.long)]
                new_coords = torch.vstack((new_coords, nc))
                new_graph_idx = torch.hstack((new_graph_idx, graph_idx[i]*torch.ones(nc.__len__())))
                new_edge_list = torch.vstack((new_edge_list, data[torch.where(data[:,-1]==i)[0]][:,:2]))
                target = torch.hstack((target, data[i,3]))
        return new_coords.to(torch.float), new_graph_idx.to(torch.long), new_edge_list.to(torch.long), target.to(torch.float)

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


def synthetic_dataset(path = 'data/', device='cpu'):
    """Synthetic Data á la Felix!
    Loads a data set that was created in XXXXX.py
    """

    extras = False

    edges = pd.read_csv(f'{path}/edgelist_synthetic.csv')
    data = torch.tensor(np.array(edges))

    coords = pd.read_csv(f'{path}/coordinates_synthetic.csv')
    graph_coords = torch.tensor(np.array(coords))

    # edge to graph index is last column ind ata
    edge_to_graph = data[:, -1]

    num_graphs = int(edge_to_graph.max()) + 1

    graph_list = range(num_graphs)
    # todo is target edge_length?
    graph_info = {'graph_list': graph_list, 'target': 'edge_lengths', 'extras': extras}

    graph_data = {}
    for graph_idx in graph_list:
        graph_data[graph_idx] = {}
        # NODE RELATED:
        #   - use node_list to filter some data further on
        graph_data[graph_idx]['num_nodes'] = len((data[torch.where(data[:, -1] == graph_idx)[0]][:, :2]).flatten().unique())

        graph_data[graph_idx]['node_list'] = torch.arange(0, graph_data[graph_idx]['num_nodes'])
        graph_data[graph_idx]['graph_idx'] = graph_idx # for reference
        # EDGE RELATED:
        graph_data[graph_idx]['node_coordinates'] = graph_coords[torch.unique(data[torch.where(data[:, -1] == graph_idx)[0], 0]).to(torch.long)].to(torch.long)
        # Edge list - fully connected graphs thus perform possible combinations
        graph_data[graph_idx]['edge_list'] = torch.tensor(list(combinations(graph_data[graph_idx]['node_list'], 2))).to(torch.long)
        #graph_data[graph_idx]['edge_list'] = data[torch.where(data[:, -1] == graph_idx)[0]][:, :2]

    return graph_info, graph_data



def QM7_dataset(path='data/qm7.mat', device='cpu'):

    # TODO: implement this?
    device = device

    # NOTE: define extras for this dataset apart from the common
    extras = ['molecule_energy', 'node_charge', 'edge_coulomb']

    # define path if necessary
    qm7 = scipy.io.loadmat(path)
    # GRAPH/MOLECULE RELATED:
    # Number of graphs in the dataset, i.e. molecules
    num_graphs = len(qm7['T'][0]) # T is atomization energies (target)
    # Graph list, each molecule has a distinct index
    graph_list = range(num_graphs)
    graph_info = {'graph_list': graph_list, 'target': 'molecule_energy', 'extras': extras}

    num_nodes = int((qm7['Z'] > 0).sum()) # used to assert later
    # looping each molecule
    graph_data = {}

    for graph_idx in graph_list:
        graph_data[graph_idx] = {}
        graph_data[graph_idx]['graph_idx'] = graph_idx # for reference

        graph_data[graph_idx]['molecule_energy'] = torch.tensor(qm7['T'][0][graph_idx])
        # from 0 and up
        # NODE/ATOM RELATED:
        #   - use node_list to filter some data further on
        graph_data[graph_idx]['node_list'] = torch.tensor(np.array(range(int((qm7['Z'][graph_idx] > 0).sum()))))
        graph_data[graph_idx]['num_nodes'] = len(graph_data[graph_idx]['node_list'])
        # Node graph index, molecule number each atom belongs to
        # Node coordinates
        graph_data[graph_idx]['node_coordinates'] = torch.tensor(qm7['R'][graph_idx])[graph_data[graph_idx]['node_list']]
        # Node atomic charge
        # total charges higher than 0 (there are no negative and 0 charged atoms, see above)
        graph_data[graph_idx]['node_charge'] = torch.tensor(qm7['Z'][graph_idx][graph_data[graph_idx]['node_list']])
        # EDGE RELATED:
        # Edge list - fully connected graphs thus perform possible combinations
        graph_data[graph_idx]['edge_list'] = torch.tensor(list(combinations(graph_data[graph_idx]['node_list'],2))).to(torch.long)
        # the coulomb value for each edge
        graph_data[graph_idx]['edge_coulomb'] = torch.zeros(len(graph_data[graph_idx]['edge_list']))
        for edge_idx, edge in enumerate(graph_data[graph_idx]['edge_list']):
            graph_data[graph_idx]['edge_coulomb'][edge_idx] = torch.tensor(qm7['X'][graph_idx][edge[0], edge[1]])
            # NOTE: GraphDataset automatically calculates edge_lengths
    return graph_info, graph_data







def get_loaders(graph_info, graph_data, batch_size=64, test_size=0.2, val_size=0.2, random_state=42, shuffle=True):
    graph_list = graph_info['graph_list']
    target = graph_info['target']
    extras = graph_info['extras']
    # splitting based on graph list
    #   - dataset is shuffled
    train_idxs, test_idxs = train_test_split(graph_list, test_size=test_size, random_state=random_state,
                                             shuffle=shuffle)
    # again, split to get val data (on train data)
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=val_size, random_state=random_state,
                                            shuffle=shuffle)
    # create a data class with __getitem__, i.e. iterable for dataloaders
    train_data = iterate_data(graph_data, train_idxs)
    val_data = iterate_data(graph_data, val_idxs)
    test_data = iterate_data(graph_data, test_idxs)

    collate_fn = collate_fn_class(extras=extras, target=target)

    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True)

    return (train_loader, val_loader, test_loader)


class collate_fn_class():
    """
    Used to collect a list/batch of graphs into a single data object, in this case GraphDataset(),
    so that it is ready to as input in a GNN.
    I.e. the goal is to collect the data for all graphs into single dtype instances
    Class object to use extra variables if necessary/defined
    """
    def __init__(self, extras=False, target='edge_length'):
        self.extras = extras
        self.target = target


    def __call__(self, data):
        # defining batch as a graph dataset
        #   - inheritance from:
        batch = GraphDataset()
        # defining variables
        batch.num_graphs = torch.tensor(len(data))
        # NOTE: graph indices are reset to 0
        # batch.graph_list = torch.zeros(batch.num_graphs)
        batch.graph_list = torch.arange(0, batch.num_graphs)
        # initialize
        batch.node_list = torch.tensor([])
        batch.node_graph_index = torch.tensor([])
        batch.node_coordinates = torch.tensor([])
        batch.edge_list = torch.tensor([])

        # if dataset has extra variables, initialize
        if self.extras != False:
            extra_data = {}
            for attribute in self.extras:
                extra_data[attribute] = torch.tensor([])

        # iterating variables in batch
        # to keep graphs distinct
        node_ref_idx = 0
        # todo: is concat efficient?
        for idx, graph_data in enumerate(data):
            # idx for current graph
            num_nodes = len(graph_data['node_list'])
            batch.node_list = torch.concat([batch.node_list, node_ref_idx + graph_data['node_list']])
            batch.node_graph_index = torch.concat([batch.node_graph_index, torch.ones(num_nodes) * idx])
            batch.node_coordinates = torch.concat([batch.node_coordinates, graph_data['node_coordinates']])
            batch.edge_list = torch.concat([batch.edge_list, node_ref_idx + graph_data['edge_list']])
            # todo extras
            if self.extras != False:
                for attribute in self.extras:
                    # concatenating data
                    #   NOTE: object has to be 1D and dtype and if index related properties are not implemented
                    extra_data[attribute] = torch.concat([extra_data[attribute], torch.unsqueeze(graph_data[attribute], dim=-1)])
                # increment to avoid duplicated (nodes have to be distinct across graphs)
            node_ref_idx += num_nodes

        # has to be integers
        batch.node_list = batch.node_list.to(torch.long)
        batch.edge_list = batch.edge_list.to(torch.long)
        batch.node_graph_index = batch.node_graph_index.to(torch.long)

        if self.extras != False:
            for attribute in self.extras:
                setattr(batch, attribute, extra_data[attribute])
        # defining target
        batch.target = getattr(batch, self.target)
        return batch




class iterate_data(Dataset):
    '''
    Used to get a graph based on its index when looping
    '''
    def __init__(self, graph_data, subset_idxs):
        # converting to tensor
        self.N = len(subset_idxs)
        # contains graph indices for current subset
        self.subset_idxs = subset_idxs
        # returns graph data given its idx
        self.graph_data = graph_data

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # extract graph index
        graph_idx = self.subset_idxs[idx]
        # return graph data
        return self.graph_data[graph_idx]



import matplotlib.pyplot as plt
import pandas as pd

class ToyDataset1D:

    def __init__(self, B: int, N: int, device: torch.device, visualize_on_load=True, seed=42):
        # Set seed and device
        self.seed = seed
        torch.manual_seed(seed)
        self.device = device
        # Create batches
        self.N = N # NUMBER OF POINTS
        self.B = B # BATCH SIZE
        self.create_dataset()
        if visualize_on_load == True:
            self.visualize_dataset(figsize=(8, 6))

        # Restructure #TODO: follow same structure as for molecular dataloader
        self.data = list(zip(self.data['data'], self.data['target']))

    def create_dataset(self):
        self.data = {}
        order_ = torch.randperm(self.N)  # shuffle data
        self.data['data'] = torch.arange(-4, 4, 8 / self.N)[order_].reshape(-1, self.B, 1).to(self.device)
        self.data['target'] = self.data['data'] ** 3 + (torch.randn(self.N, 1).reshape(-1, self.B, 1) * 3).to(self.device)

    def visualize_dataset(self, figsize=(12, 8)):
        # plot data
        plt.figure(figsize=figsize)
        plt.plot(self.data['data'].detach().flatten().cpu(), self.data['target'].detach().flatten().cpu(), 'k.')
        plt.plot(torch.arange(-4, 4, 8 / self.N), torch.arange(-4, 4, 8 / self.N) ** 3, 'r--')
        plt.xlim([-6.5, 6.5])
        plt.show()

    def plot_regression_line(self, model: torch.nn.Module, plot_uncertainty=True, save_path=None, show=True):
        # Predict on the data range
        toy_ = torch.arange(-6, 6, 12 / self.N).reshape(-1, 1).to(self.device)
        outputs = model(toy_)

        # Get evidential parameters
        gamma, v, alpha, beta = torch.tensor_split(outputs, 4, axis=1)

        # Reformat tensors
        xaxis = toy_.detach().flatten().cpu().numpy()
        y_true = (toy_ ** 3).detach().flatten().cpu().numpy()
        y_pred = gamma.detach().flatten().cpu().numpy()
        aleatoric = (beta / (alpha - 1)).detach().flatten().cpu().numpy()
        epistemic = (beta / (v * (alpha - 1))).detach().flatten().cpu().numpy()

        # Gather information in dataframe for plotting
        results = pd.DataFrame({'xaxis': xaxis,
                                'y_true': y_true,
                                'y_pred': y_pred,
                                'aleatoric': aleatoric,
                                'epistemic': epistemic})

        # Plot regression line

        plt.plot(results['xaxis'], results['y_true'], '--r')
        plt.plot(results['xaxis'], results['y_pred'])

        plt.vlines(-4, -6.5 ** 3, 6.5 ** 3, colors='gray', linestyles='--')
        plt.vlines(4, -6.5 ** 3, 6.5 ** 3, colors='gray', linestyles='--')
        plt.fill_betweenx(pd.Series(np.arange(-6.5 ** 3, 6.5 ** 3)), -6, -4, alpha=.3, interpolate=True, color='gray')
        plt.fill_betweenx(pd.Series(np.arange(-6.5 ** 3, 6.5 ** 3)), 4, 6, alpha=.3, interpolate=True, color='gray')

        if plot_uncertainty == True:
            plt.fill_between(results['xaxis'], results['y_pred'] - results['epistemic'],
                             results['y_pred'] + results['epistemic'], alpha=.3, interpolate=True)  # step='post')

        plt.xlim([-6.5, 6.5])
        plt.ylim([-6.5 ** 3, 6.5 ** 3])

        if save_path != None:
            plt.savefig(save_path)
        if show == True:
            plt.show()

