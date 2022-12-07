import os

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import scipy.io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from tqdm import tqdm
from numba import jit

from dgl.data import QM9Dataset


class ToyDataset1D:
    def __init__(self, B: int, N: int, range_, device: torch.device, noise_level: float, visualize_on_load=False,
                 seed=42):
        # Set seed and device
        self.seed = seed
        torch.manual_seed(seed)
        self.device = device
        self.range_ = range_
        self.noise_level = noise_level

        # Create batches
        self.N = N  # NUMBER OF POINTS
        self.B = B  # BATCH SIZE
        self.num_batches = N // B  # NUMBER OF BATCHES

        self.create_dataset()
        if visualize_on_load == True:
            self.visualize_dataset(figsize=(8, 6))

        class BATCH:  # D
            def __init__(self, data_, target_):
                self.data = data_
                self.target = target_

        # Restructure
        self.batches = []
        for batch_idx in range(self.num_batches):
            batch_data_ = self.data['data'][batch_idx, :].view(B, -1)
            batch_targets_ = self.data['target'][batch_idx, :].view(B, -1)
            self.batches.append(BATCH(batch_data_, batch_targets_))

        self.unbatched_data = self.data
        self.data = list(zip(self.data['data'], self.data['target']))

    def create_dataset(self, ):
        self.data = {}
        order_ = torch.randperm(self.N)  # shuffle data
        self.data['data'] = torch.FloatTensor(self.N).uniform_(self.range_[0], self.range_[1])[order_].reshape(-1,
                                                                                                               self.B,
                                                                                                               1).to(
            self.device)
        self.data['target'] = self.data['data'] ** 3 + (
                    torch.randn(self.N, 1).reshape(-1, self.B, 1) * self.noise_level).to(self.device)

    def visualize_dataset(self, ):
        # plot data
        plt.figure()
        plt.plot(self.unbatched_data['data'].detach().flatten().cpu(),
                 self.unbatched_data['target'].detach().flatten().cpu(), 'ko', markersize=0.5)
        plt.plot(torch.arange(-6, 6, 12 / self.N), torch.arange(-6, 6, 12 / self.N) ** 3, 'r--')

        plt.vlines(-4, -6.5 ** 3, 6.5 ** 3, colors='gray', linestyles='--')
        plt.vlines(4, -6.5 ** 3, 6.5 ** 3, colors='gray', linestyles='--')
        plt.fill_betweenx(pd.Series(np.arange(-6.5 ** 3, 6.5 ** 3)), -6.5, -4, alpha=.3, interpolate=True, color='gray')
        plt.fill_betweenx(pd.Series(np.arange(-6.5 ** 3, 6.5 ** 3)), 4, 6.5, alpha=.3, interpolate=True, color='gray')

        plt.xlim([-6.5, 6.5])
        plt.ylim([-150, 150])
        plt.show()

    def plot_regression_line(self, model: torch.nn.Module, epoch: int, uncertainty_types: list, save_path=None,
                             show=False):
        plt.style.use('ggplot')

        # Predict on the data range
        toy_ = torch.arange(-6, 6, 12 / (4 * self.N)).reshape(-1, 1).to(self.device)
        outputs = model(toy_)

        # Reformat tensors
        xaxis = toy_.detach().flatten().cpu().numpy()
        y_true = (toy_ ** 3).detach().flatten().cpu().numpy()

        # Get evidential parameters
        if 'Evidential' in model.__class__.__name__:
            gamma, nu, alpha, beta = torch.tensor_split(outputs, 4, axis=1)

            fig, ax = plt.subplots(2, 2, sharex=True)
            ax[0, 0].plot(xaxis, gamma.cpu())
            ax[0, 0].set_title(r'$\gamma$')
            ax[0, 1].plot(xaxis, nu.cpu())
            ax[0, 1].set_title(r'$\nu$')
            ax[1, 0].plot(xaxis, alpha.cpu())
            ax[1, 0].set_title(r'$\alpha$')
            ax[1, 1].plot(xaxis, beta.cpu())
            ax[1, 1].set_title(r'$\beta$')
            ax[0, 0].set_xticks(np.arange(-6, 7, 2), np.arange(-6, 7, 2))

        else:
            gamma, sigma = torch.tensor_split(outputs, 2, axis=1)

            fig, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 8))
            ax[0].plot(xaxis, gamma.cpu())
            ax[0].set_title(r'$\mu$')
            ax[1].plot(xaxis, sigma.cpu())
            ax[1].set_title(r'$\sigma$')
            ax[0].set_xticks(np.arange(-6, 7, 2), np.arange(-6, 7, 2))

        fig.suptitle(f"EPOCH {epoch}")
        fig.suptitle(f"PARAMETERS (EPOCH = {epoch})", fontsize=20, weight='bold')
        os.makedirs(save_path / 'PARAMS', exist_ok=True)
        plt.savefig(save_path / f"PARAMS/0{epoch}.png")

        if show == True:
            fig.show()
        plt.close("all")

        # Determine predicted output
        y_pred = gamma.detach().flatten().cpu().numpy()
        # Gather information in dataframe for plotting
        results = pd.DataFrame({'xaxis': xaxis,
                                'y_true': y_true,
                                'y_pred': y_pred, })

        for uncertainty_type in uncertainty_types:
            os.makedirs(save_path / uncertainty_type.upper(), exist_ok=True)

            # Print uncertainty estimates
            if uncertainty_type == 'aleatoric':
                uncertainty = (beta / (alpha - 1)).detach().flatten().cpu().numpy()
            elif uncertainty_type == 'epistemic':
                uncertainty = (beta / (nu * (alpha - 1))).detach().flatten().cpu().numpy()
            else:
                uncertainty = sigma.detach().flatten().cpu().numpy()

            results[f'{uncertainty_type}'] = uncertainty  # Store uncertainty
            results[uncertainty_type].replace(np.inf, 1e6, inplace=True)  # Replace inf for visualization purposes

            # Print uncertainty estimates (mainly for debugging...)
            # print(f"\n{uncertainty_type.upper()} (-4, 4): {np.round(results[np.logical_and(results['xaxis'] > -4, results['xaxis'] < 4)][uncertainty_type].sum(), 2)}")
            # print(f"{uncertainty_type.upper()} ]-4, 4[: {np.round(results[np.logical_or(results['xaxis'] < -4, results['xaxis'] > 4)][uncertainty_type].sum(), 2)}")

            # Plot regression line and data points
            plt.figure(dpi=250)
            plt.plot(self.unbatched_data['data'].detach().flatten().cpu(),
                     self.unbatched_data['target'].detach().flatten().cpu(), 'ko', markersize=0.5)
            plt.plot(results['xaxis'], results['y_true'], '--r')
            plt.plot(results['xaxis'], results['y_pred'], color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])

            plt.vlines(-4, -6.5 ** 3, 6.5 ** 3, colors='gray', linestyles='--')
            plt.vlines(4, -6.5 ** 3, 6.5 ** 3, colors='gray', linestyles='--')
            plt.fill_betweenx(pd.Series(np.arange(-6.5 ** 3, 6.5 ** 3)), -6.5, -4, alpha=.3, interpolate=True,
                              color='gray')
            plt.fill_betweenx(pd.Series(np.arange(-6.5 ** 3, 6.5 ** 3)), 4, 6.5, alpha=.3, interpolate=True,
                              color='gray')

            plt.fill_between(results['xaxis'], results['y_pred'] - 3 * np.sqrt(results[uncertainty_type]),
                             results['y_pred'] + 3 * np.sqrt(results[uncertainty_type]), alpha=.2, interpolate=True,
                             color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])  # step='post')
            plt.fill_between(results['xaxis'], results['y_pred'] - 2 * np.sqrt(results[uncertainty_type]),
                             results['y_pred'] + 2 * np.sqrt(results[uncertainty_type]), alpha=.2, interpolate=True,
                             color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])  # step='post')
            plt.fill_between(results['xaxis'], results['y_pred'] - 1 * np.sqrt(results[uncertainty_type]),
                             results['y_pred'] + 1 * np.sqrt(results[uncertainty_type]), alpha=.2, interpolate=True,
                             color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])  # step='post')

            plt.xlim([-6.5, 6.5])
            plt.ylim([-150, 150])
            plt.title(f"{uncertainty_type.upper()} (EPOCH = {epoch})", fontsize=20, weight='bold')

            if save_path != None:
                plt.savefig(save_path / f"{uncertainty_type}/0{epoch}.png")
            if show == True:
                # plt.show(block=False)
                plt.close("all")
            plt.close("all")

        if 'aleatoric' in uncertainty_types and 'epistemic' in uncertainty_types and show == True:
            os.makedirs(save_path / "COMBINED_UNCERTAINTIES", exist_ok=True)
            results.plot(x='xaxis', y=['aleatoric', 'epistemic'])
            plt.savefig(save_path / f"COMBINED_UNCERTAINTIES/0{epoch}.png")
            plt.close("all")


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
            self.node_coordinates[self.node_graph_index == i] = coords - mean
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
                                                                        coords - mean) @ R + mean
        return self


class QM9:
    """Molecular dataset.


    """

    def __init__(self, label_attr: str, num_graphs=None, test_size=0.1, device='cpu'):
        # super().__init__(dim=)

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

        return node_coordinates, node_graph_index.to(torch.long), edge_list.to(torch.long), torch.tensor(res).to(
            torch.float)


def synthetic_dataset(path='data/', device='cpu'):
    """Synthetic Data á la Felix!
    Loads a data set that was created in XXXXX.py
    """

    extras = ['molecule_summed_force']

    edges = pd.read_csv(f'{path}edgelist.csv')
    data = torch.tensor(np.array(edges))

    coords = pd.read_csv(f'{path}coords.csv')
    graph_coords = torch.tensor(np.array(coords))

    # edge to graph index is last column ind ata
    edge_to_graph = data[:, -1]

    num_graphs = int(edge_to_graph.max()) + 1

    graph_list = range(num_graphs)
    # todo is target edge_length?
    graph_info = {'graph_list': graph_list, 'target': 'molecule_summed_force', 'extras': extras}

    graph_data = {}
    for graph_idx in graph_list:
        graph_data[graph_idx] = {}
        # NODE RELATED:
        #   - use node_list to filter some data further on
        graph_data[graph_idx]['num_nodes'] = len(
            (data[torch.where(data[:, -1] == graph_idx)[0]][:, :2]).flatten().unique())

        graph_data[graph_idx]['node_list'] = torch.arange(0, graph_data[graph_idx]['num_nodes']).to(
            torch.device(device))
        graph_data[graph_idx]['graph_idx'] = graph_idx  # for reference
        # EDGE RELATED:
        graph_data[graph_idx]['node_coordinates'] = graph_coords[
            torch.unique(data[torch.where(data[:, -1] == graph_idx)[0], 0]).to(torch.long)].double().to(
            torch.device(device))
        # Edge list - fully connected graphs thus perform possible combinations
        graph_data[graph_idx]['edge_list'] = torch.tensor(list(combinations(graph_data[graph_idx]['node_list'], 2))).to(
            torch.long).to(torch.device(device))
        # graph_data[graph_idx]['edge_list'] = data[torch.where(data[:, -1] == graph_idx)[0]][:, :2]
        graph_data[graph_idx]['molecule_summed_force'] = data[torch.where(data[:, -1] == graph_idx)[0]][:, 3][0]
    return graph_info, graph_data


def QM7_dataset(path, device=torch.device('cpu')):
    # NOTE: define extras for this dataset apart from the common
    extras = ['molecule_energy', 'node_charge', 'edge_coulomb']

    # define path if necessary
    qm7 = scipy.io.loadmat(path)
    # GRAPH/MOLECULE RELATED:
    # Number of graphs in the dataset, i.e. molecules
    num_graphs = len(qm7['T'][0])  # T is atomization energies (target)
    # Graph list, each molecule has a distinct index
    graph_list = range(num_graphs)
    graph_info = {'graph_list': graph_list, 'target': 'molecule_energy', 'extras': extras}

    num_nodes = int((qm7['Z'] > 0).sum())  # used to assert later

    # looping each molecule
    graph_data = {}
    for graph_idx in tqdm(graph_list):
        graph_data[graph_idx] = {}
        graph_data[graph_idx]['graph_idx'] = graph_idx  # for reference

        graph_data[graph_idx]['molecule_energy'] = torch.tensor(qm7['T'][0][graph_idx]).to(device)
        # from 0 and up
        # NODE/ATOM RELATED:
        #   - use node_list to filter some data further on
        graph_data[graph_idx]['node_list'] = torch.tensor(np.array(range(int((qm7['Z'][graph_idx] > 0).sum())))).to(
            device)
        graph_data[graph_idx]['num_nodes'] = len(graph_data[graph_idx]['node_list'])
        # Node graph index, molecule number each atom belongs to
        # Node coordinates
        graph_data[graph_idx]['node_coordinates'] = torch.tensor(qm7['R'][graph_idx]).to(device)[
            graph_data[graph_idx]['node_list'].to(torch.long).to(device)]
        # Node atomic charge
        # total charges higher than 0 (there are no negative and 0 charged atoms, see above)
        graph_data[graph_idx]['node_charge'] = torch.tensor(qm7['Z'][graph_idx]).to(device)[
            graph_data[graph_idx]['node_list'].to(torch.long)]
        # EDGE RELATED:
        # Edge list - fully connected graphs thus perform possible combinations
        graph_data[graph_idx]['edge_list'] = torch.tensor(list(combinations(graph_data[graph_idx]['node_list'], 2))).to(
            torch.long).to(device)
        # the coulomb value for each edge
        graph_data[graph_idx]['edge_coulomb'] = torch.zeros(len(graph_data[graph_idx]['edge_list'])).to(device)
        for edge_idx, edge in enumerate(graph_data[graph_idx]['edge_list']):
            graph_data[graph_idx]['edge_coulomb'][edge_idx] = torch.tensor(qm7['X'][graph_idx][edge[0], edge[1]]).to(
                device)
            # NOTE: GraphDataset automatically calculates edge_lengths
    return graph_info, graph_data


def get_loaders(graph_info, graph_data, device, batch_size=64, test_size=0.2, val_size=0.2, random_state=42,
                shuffle=True):
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

    collate_fn = collate_fn_class(extras=extras, target=target, device=device)

    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


class collate_fn_class():
    """
    Used to collect a list/batch of graphs into a single data object, in this case GraphDataset(),
    so that it is ready to as input in a GNN.
    I.e. the goal is to collect the data for all graphs into single dtype instances
    Class object to use extra variables if necessary/defined
    """

    def __init__(self, extras=False, target='edge_length', device=torch.device('cpu')):
        self.extras = extras
        self.target = target
        self.device = device

    def __call__(self, data):
        # defining batch as a graph dataset
        #   - inheritance from:
        batch = GraphDataset()
        # defining variables
        batch.num_graphs = torch.tensor(len(data)).to(self.device)
        # NOTE: graph indices are reset to 0
        # batch.graph_list = torch.zeros(batch.num_graphs)
        batch.graph_list = torch.arange(0, batch.num_graphs).to(self.device)
        # initialize
        batch.node_list = torch.tensor([]).to(self.device)
        batch.node_graph_index = torch.tensor([]).to(self.device)
        batch.node_coordinates = torch.tensor([]).to(self.device)
        batch.edge_list = torch.tensor([]).to(self.device)

        # if dataset has extra variables, initialize
        if self.extras != False:
            extra_data = {}
            for attribute in self.extras:
                extra_data[attribute] = torch.tensor([]).to(self.device)

        # iterating variables in batch
        # to keep graphs distinct
        node_ref_idx = 0
        # todo: is concat efficient?
        for idx, graph_data in enumerate(data):
            # idx for current graph
            num_nodes = len(graph_data['node_list'])
            batch.node_list = torch.concat([batch.node_list, node_ref_idx + graph_data['node_list']])
            batch.node_graph_index = torch.concat([batch.node_graph_index, torch.ones(num_nodes).to(self.device) * idx])
            batch.node_coordinates = torch.concat([batch.node_coordinates, graph_data['node_coordinates']])
            batch.edge_list = torch.concat([batch.edge_list, node_ref_idx + graph_data['edge_list']])
            # todo extras
            if self.extras != False:
                for attribute in self.extras:
                    # concatenating data
                    #   NOTE: object has to be 1D and dtype and if index related properties are not implemented
                    extra_data[attribute] = torch.concat(
                        [extra_data[attribute], torch.unsqueeze(graph_data[attribute], dim=-1)])
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