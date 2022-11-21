
def write_to_tensorboard(writer, loss):
    pass


def cross2D(v1, v2):
    """Compute the 2-d cross product.

    Parameters
    ----------
    v1, v2 : Array
        Arrays (shape Nx2) containing N 2-d vectors

    Returns
    -------
    Array
        Array (shape N) containing the cross products

    """
    return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

def dot2D(v1, v2):
    """Compute the 2-d dot product.

    Parameters
    ----------
    v1, v2 : Array
        Arrays (Nx2) containing N 2-d vectors

    Returns
    -------
    Array
        Array (shape N) containing the dot products

    """
    return v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]

def dot3D(v1, v2):
    """Compute the 2-d dot product.

    Parameters
    ----------
    v1, v2 : Array
        Arrays (Nx2) containing N 3-d vectors

    Returns
    -------
    Array
        Array (shape N) containing the dot products

    """
    return v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1] + v1[:, 2] * v2[:, 2]

import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import trange



def training(loaders, net, optimizer, lr=0.01, loss_function=torch.nn.MSELoss(), epochs=1000,
             val_every_step=50, store_train_every=50,
             tensorboard_logdir='logs', experiment_name=str(time.time())):

    # Setup tensorboard
    writer = SummaryWriter(Path(f"{tensorboard_logdir}")/experiment_name)

    # unpacking loaders
    train_loader, val_loader, test_loader = loaders

    # Setup parameters
    net.train()
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Setup storage information
    train_lss, val_lss = np.zeros(epochs), np.zeros(epochs // val_every_step)
    train_acc, val_acc = np.zeros(epochs), np.zeros(epochs // val_every_step)

    with trange(epochs) as t:

        val_step = 0
        for epoch in t:
            for inx_batch, train_batch in enumerate(train_loader):
                # Run forward pass
                optimizer.zero_grad()
                outputs = net(train_batch)

                # Compute loss
                #loss = loss_function(outputs, train_batch.target)
                loss.backward()
                optimizer.step()

                gamma, v, alpha, beta = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]

                loss, (nll, reg_loss) = loss_function(train_batch.target, gamma, v, alpha, beta, return_comps=True)

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
                        for inx_batch, val_batch in enumerate(train_loader):

                            outputs = net(val_batch)
                            loss = loss_function(outputs, val_batch.target)
                            #loss, (nll, reg_loss) = loss_function(y, mu, v, alpha, beta, return_comps=True)


                        # Store validation losses
                        writer.add_scalar('Loss/validation', loss, epoch)
                        val_lss[val_step] = loss
                        val_acc[val_step] = torch.mean(torch.pow(outputs - val_batch.target, 2))
                        net.train()

                    # Print status
                    t.set_description_str(f'Training Loss: {train_lss[epoch]:.3f} \t| \t Validation Loss: {val_lss[val_step]:.3f} | Progress')
                    print("")

                    # Update validation index
                    val_step += 1

                # todo save best performing model?
    # Close tensorboard
    writer.close()


