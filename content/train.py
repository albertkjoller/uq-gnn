
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from tqdm import trange

import Datasets
from GNNModels import EvidentialGNN3D, GNNInvariant
from losses import NIGLoss, MSELoss

from losses import NIGLoss
from GNNModels import EvidentialGNN3D
from losses import NIGLoss, MSELoss
from GNNModels import GNNInvariant

from utils import training
from Datasets import *


def train(train_data, test_data, net, optimizer, lr=0.01, loss_function=torch.nn.MSELoss(), epochs=1000,
          val_every_step=50, store_train_every=50,
          tensorboard_logdir='logs_felix', experiment_name=str(time.time())):

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
            optimizer.zero_grad()
            # Run forward pass
            outputs = net(train_data)

            # Compute loss
            (loss_name, loss), xtra_losses = loss_function(outputs, train_data.target)
            loss.backward()
            optimizer.step()

            # Store train losses
            if epoch % store_train_every == 0:
                train_lss[epoch] = loss

                writer.add_scalar(f'TRAIN/{loss_name}', loss, epoch)
                if xtra_losses != None:
                    for name, loss_ in xtra_losses:
                        writer.add_scalar(f'TRAIN/{name}', loss_, epoch)

            # Validation run
            if epoch % val_every_step == 0:
                with torch.no_grad():
                    # Change network mode
                    net.eval()
                    outputs = net(test_data)

                    # Compute loss
                    (loss_name, loss), xtra_losses = loss_function(outputs, test_data.target)

                    val_lss[val_step] = loss
                    writer.add_scalar(f'VALIDATION/{loss_name}', loss, epoch)
                    if xtra_losses != None:
                        for name, loss_ in xtra_losses:
                            writer.add_scalar(f'VALIDATION/{name}', loss_, epoch)

                    net.train()

                # Print status
                t.set_description_str(f'Training Loss: {train_lss[epoch]:.3f} \t| \t Validation Loss: {val_lss[val_step]:.3f} | Progress')
                print("")

                # Update validation index
                val_step += 1

    # Close tensorboard
    writer.close()

if __name__ == '__main__':

    seed = 42
    # Training parameters
    epochs = 500
    learning_rate = 0.01
    batch_size = 32
    # testing parameters
    store_train_every = 50
    val_every_step = 50
    # Tensorboard configurations
    tensorboard_logdir = '../logs'
    experiment_name = 'test2'
    # dataset parameters
    shuffle = True
    # Utilize GPU?
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load data
    #QM9 = Datasets.QM9(label_attr='U0', num_graphs=50, device=device)
    #train_data = QM9.data['train']
    #test_data = QM9.data['test']


    # Load data
    graph_info, graph_data = synthetic_dataset(path='data', device=device)
    #graph_info, graph_data = QM7_dataset(path='data/qm7.mat', device = device)

    loaders = get_loaders(graph_info, graph_data, batch_size=batch_size, test_size=0.2, val_size=0.2, random_state=seed, shuffle=shuffle)


    #dataset = Datasets.QM9(label_attr='U0', num_graphs=100, device=device)
    #train_data = dataset.data['train']
    #test_data = dataset.data['test']
    # Setup regression network

    net = EvidentialGNN3D(state_dim=10, num_message_passing_rounds=3, device=device)
    loss_function = MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),)
    # Run training loop
    training(loaders, net, optimizer, loss_function=loss_function, lr=learning_rate,
          epochs=epochs, val_every_step=val_every_step, store_train_every=store_train_every,
          experiment_name=experiment_name, tensorboard_logdir=tensorboard_logdir)

    # TODO: modify for edge_dim=2 (if e.g. coulomb matrix and distance?)


    # Training parameters
    epochs = 4000

    # Tensorboard configurations
    tensorboard_logdir = 'logs_felix/'
    experiment_name = 'evidential_test_2'


    # Run training loop
    #seed 0 is good, 42 is bad!
    torch.manual_seed(42)
    train(train_data, test_data, net, optimizer, loss_function=loss_function, lr=0.01,
          epochs=epochs, val_every_step=50, store_train_every=50,
          experiment_name=experiment_name, tensorboard_logdir=tensorboard_logdir)
    #train(train_data, test_data, net, optimizer, loss_function=loss_function, lr=0.01,
    #      epochs=epochs, val_every_step=50, store_train_every=50,
    #net = GNNInvariant(state_dim=10, output_dim=4, num_message_passing_rounds=3, device=device)
    #loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),)


    #train(train_data, test_data, net, optimizer, loss_function=loss_function, lr=0.01,
    #      epochs=500, val_every_step=50, store_train_every=50,
    #      experiment_name=experiment_name, tensorboard_logdir=tensorboard_logdir)



