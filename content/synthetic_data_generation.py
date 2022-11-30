'''
This file contains code for generation of synthetic data
'''

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

def U(x):
    '''
    As far as we understand, this is the force between two atoms. We found it here:
    https://www.toppr.com/ask/question/the-potential-energy-function-for-the-force-between-two-atoms/
    We take as input the distance between two atoms and compute as output the
    force between the two atoms. What is important here is that this relation is non-linear!
    :param: x, is the distance between two atoms
    :output: the force between two atoms
    '''
    a = 500000
    b = 4000
    return ((a/(x**12)) - (b/(x**6)))


def get_molecule(mol_size=4, node_nr_start=0, seed=42, center_box=(-5,5)):
    '''
    Takes molecule size and number of molecules and returns an edgelist with n_mols number of
    disconnected cliques of size mol_size. If we want to run this multiple times, calculate the number of nodes
    in the graph already and set node_nr_start to that.
    :param mol_size: Size of molecules (nr. of atoms)
    :param node_nr_start: Number of nodes in the graph before running this function
    :return: edges (a weighted edgelist of the graph)
    '''
    np.random.seed(seed)
    inp1 = np.ones(mol_size,dtype=int)
    blob = torch.tensor(make_blobs(inp1, n_features=3,center_box=center_box, cluster_std=0.1)[0])
    edges = []
    predictor = []
    for source in range(blob.shape[0]):
        k_neighbours = torch.argsort(torch.sqrt((blob[source].unsqueeze(0) - blob)**2).sum(1))[:mol_size]
        k_neighbours = k_neighbours[1:]
        weights = torch.sort(torch.sqrt(((blob[source].unsqueeze(0)-blob)**2).sum(1)))[0][:mol_size]
        weights = weights[1:]
        force = 0
        for idx, target in enumerate(k_neighbours):
            edges.append((torch.tensor(source+node_nr_start),target+node_nr_start,weights[idx]))
            force += U(weights[idx]) #We use a force or energy function as a predictor!
        predictor.append(force)

    edges = torch.tensor([[col1, col2, col3] for ((col1, col2, col3)) in edges])
    predictor = torch.zeros(len(edges)) + torch.sum(torch.tensor(predictor))
    edges = torch.hstack((edges,predictor.unsqueeze(1)))
    graph_coords = torch.zeros(len(edges)) + torch.tensor(seed)
    edges = torch.hstack((edges, graph_coords.unsqueeze(1)))
    node_nr_end = int(edges.max(0)[0][0])+1
    return edges, node_nr_end, blob


def create_molecules(mol_sizes, filename, center_box=(-5, 5)):
    for idx, mol_size in enumerate(mol_sizes):
        if idx == 0:
            edges, node_nr_end, coords = get_molecule(int(mol_size), node_nr_start=idx, seed=idx, center_box=center_box)
            coordinates = coords
            edge_list = edges
        else:
            edges, node_nr_end, coords = get_molecule(int(mol_size), node_nr_start=node_nr_end, seed=idx,
                                                      center_box=center_box)
            coordinates = torch.vstack((coordinates, coords))
            edge_list = torch.vstack((edge_list, edges))

    edge_np = edge_list.numpy()  # convert to Numpy array
    df = pd.DataFrame(edge_np)  # convert to a dataframe
    df.to_csv(f'{filename}_edgelist.csv', index=False)  # save to file

    coords_np = coordinates.numpy()  # convert to Numpy array
    df_coords = pd.DataFrame(coords_np)  # convert to a dataframe
    df_coords.to_csv(f'{filename}_coords.csv', index=False)  # save to file

"""
Below we create a training set where nodes are sampled with a centerbox of -5,5. After that we create two OOD
test sets; the first test set, the nodes are sampled with a centerbox of -0.5,0.5 and the in second
test set, the nodes are sampled with a centerbox of -10,10.
"""
if __name__== '__main__':
    mol_sizes = np.hstack((np.zeros(20)+3,np.zeros(20)+4))
    mol_sizes = np.hstack((mol_sizes, np.zeros(40)+5))

    #centerbox = (-5,5):
    create_molecules(mol_sizes,'temp_synth/2', center_box=(-2,2))

    #centerbox = (-0.5, 0.5):
    create_molecules(mol_sizes,'temp_synth/0.5', center_box=(-0.5,0.5))

    #centerbox = (-100,100):
    create_molecules(mol_sizes,'temp_synth/10', center_box=(-10,10))

