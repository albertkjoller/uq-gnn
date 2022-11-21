
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import trange

import Datasets
from GNNModels import EvidentialGNN3D
from losses import NIGLoss, MSELoss

def train(train_data, test_data, net, optimizer, loss_function, lr=0.01, epochs=1000,
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

    # Utilize GPU?
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load data
    QM9 = Datasets.QM9(label_attr='U0', num_graphs=50, device=device)
    train_data = QM9.data['train']
    test_data = QM9.data['test']

    # Setup regression network
    net = EvidentialGNN3D(state_dim=10, num_message_passing_rounds=3, device=device)
    loss_function = MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),)

    # Training parameters
    epochs = 500

    # Tensorboard configurations
    tensorboard_logdir = '../logs/'
    experiment_name = 'P_F_test'

    # Run training loop
    train(train_data, test_data, net, optimizer, loss_function=loss_function, lr=0.01,
          epochs=3000, val_every_step=50, store_train_every=50,
          experiment_name=experiment_name, tensorboard_logdir=tensorboard_logdir)





