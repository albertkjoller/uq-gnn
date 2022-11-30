'''
This file contains code for generation of synthetic data
'''

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

def get_molecule(mol_size=4, node_nr_start=0, seed=42):
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
    blob = torch.tensor(make_blobs(inp1, 3)[0])
    edges = []
    predictor = []
    for source in range(blob.shape[0]):
        k_neighbours = torch.argsort(torch.sqrt((blob[source].unsqueeze(0) - blob)**2).sum(1))[:mol_size]
        k_neighbours = k_neighbours[1:]
        weights = torch.sort(torch.sqrt((blob[source].unsqueeze(0) - blob)**2).sum(1))[0][:mol_size]
        weights = weights[1:]
        predictor.append(weights.sum(0))
        for idx, target in enumerate(k_neighbours):
            edges.append((torch.tensor(source+node_nr_start),target+node_nr_start,weights[idx]))

    edges = torch.tensor([[col1, col2, col3] for ((col1, col2, col3)) in edges])
    predictor = torch.zeros(len(edges)) + torch.sum(torch.tensor(predictor))
    edges = torch.hstack((edges,predictor.unsqueeze(1)))
    graph_coords = torch.zeros(len(edges)) + torch.tensor(seed)
    edges = torch.hstack((edges, graph_coords.unsqueeze(1)))
    node_nr_end = int(edges.max(0)[0][0])+1
    return edges, node_nr_end, blob

mol_sizes = np.hstack((np.zeros(20)+3,np.zeros(20)+4))
mol_sizes = np.hstack((mol_sizes, np.zeros(40)+5))

for idx, mol_size in enumerate(mol_sizes):
    if idx == 0:
        edges, node_nr_end, coords = get_molecule(int(mol_size),node_nr_start=idx, seed=idx)
        coordinates = coords
        edge_list = edges
    else:
        edges, node_nr_end, coords = get_molecule(int(mol_size),node_nr_start=node_nr_end, seed=idx)
        coordinates = torch.vstack((coordinates, coords))
        edge_list = torch.vstack((edge_list, edges))

edge_np = edge_list.numpy() #convert to Numpy array
df = pd.DataFrame(edge_np) #convert to a dataframe
df.to_csv("datasets_files/edgelist_synthetic.csv",index=False) #save to file

#Then, to reload:
df = pd.read_csv("datasets_files/edgelist_synthetic.csv")

coords_np = coordinates.numpy() #convert to Numpy array
df_coords = pd.DataFrame(coords_np) #convert to a dataframe
df_coords.to_csv("datasets_files/coordinates_synthetic.csv",index=False) #save to file

#Then, to reload:
#df = pd.read_csv("synthetic_data/edgelist_synthetic.csv")


'''blob_temp = blobs[0][0]
colors = ['blue','red','green','pink']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for blob_idx in range(len(blobs)):
    ax.scatter(blobs[blob_idx][0][:,0],blobs[blob_idx][0][:,1],blobs[blob_idx][0][:,2], c=colors[blob_idx])
plt.clf()
'''
