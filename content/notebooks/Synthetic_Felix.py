import torch
import networkx as nx

import pandas as pd

import sys

# setting path
sys.path.append('../')
from Datasets import SyntheticData

import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import trange

from GNNModels import GNNInvariant

def train(train_data, test_data, net, optimizer, lr=0.01, loss_function=torch.nn.MSELoss(), epochs=1000,
          val_every_step=50, store_train_every=50,
          tensorboard_logdir='logs', experiment_name=str(time.time())):

    # Setup tensorboard
    writer = SummaryWriter(Path(f"{tensorboard_logdir}")/experiment_name)

    # Setup parameters
    net.train()
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Setup storage information
    train_lss, val_lss = np.zeros(epochs), np.zeros(epochs // val_every_step)
    train_acc, val_acc = np.zeros(epochs), np.zeros(epochs // val_every_step)

    with trange(epochs) as t:

        val_step = 0
        for epoch in t:             # TODO: INCLUDE BATCH LOOP?
            # Run forward pass
            optimizer.zero_grad()
            outputs = net(train_data)

            # Compute loss
            loss = loss_function(outputs, train_data.target)
            loss.backward()
            optimizer.step()

            gamma, v, alpha, beta = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]

            # loss, (nll, reg_loss) = loss_function(train_data.target, gamma, v, alpha, beta, return_comps=True)

            # Store train losses
            if epoch % store_train_every == 0:
                writer.add_scalar('Loss/train', loss, epoch)
                train_lss[epoch] = loss
                train_acc[epoch] = torch.mean(torch.pow(outputs - train_data.target, 2))

            # Validation run
            if epoch % val_every_step == 0:
                with torch.no_grad():
                    # Change network mode
                    net.eval()
                    outputs = net(test_data)
                    loss = loss_function(outputs, test_data.target)
                    #loss, (nll, reg_loss) = loss_function(y, mu, v, alpha, beta, return_comps=True)


                    # Store validation losses
                    writer.add_scalar('Loss/validation', loss, epoch)
                    val_lss[val_step] = loss
                    val_acc[val_step] = torch.mean(torch.pow(outputs - test_data.target, 2))
                    net.train()

                # Print status
                t.set_description_str(f'Training Loss: {train_lss[epoch]:.3f} \t| \t Validation Loss: {val_lss[val_step]:.3f} | Progress')
                print("")

                # Update validation index
                val_step += 1

    # Close tensorboard
    writer.close()

if __name__ == '__main__':
    edges = pd.read_csv('../datasets_files/edgelist_synthetic.csv')
    edges = torch.tensor(np.array(edges))

    coords = pd.read_csv('../datasets_files/coordinates_synthetic.csv')
    coords = torch.tensor(np.array(coords))
    #torch.manual_seed(42)

    graph_idx = edges[:, -1]

    # Load data
    synthetic_data = SyntheticData(data=edges, graph_idx=graph_idx, graph_coords=coords)

    train_data = synthetic_data.data

    # Setup regression network
    net = GNNInvariant(state_dim=10, output_dim=4, num_message_passing_rounds=3)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),)

    # Training parameters
    epochs = 500

    # Tensorboard configurations
    tensorboard_logdir = '../logs'
    experiment_name = 'test2'

    # Run training loop
    train(train_data, train_data, net, optimizer, loss_function=loss_function, lr=0.01,
          epochs=500, val_every_step=50, store_train_every=50,
          experiment_name=experiment_name, tensorboard_logdir=tensorboard_logdir)








