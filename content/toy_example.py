
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import trange

from GNNModels import EvidentialToyModel1D
from losses import NIGLoss


def train(dataset, model, optimizer, loss_function, lr=0.01,
          epochs=1000,  device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
          store_train_every=50, val_every_step=50, tensorboard_logdir='logs', experiment_name=str(time.time())):

    # Setup tensorboard
    writer = SummaryWriter(Path(f"{tensorboard_logdir}")/experiment_name)

    # Setup parameters
    model.train()
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Setup storage information
    train_lss = np.zeros(epochs)
    train_acc = np.zeros(epochs)

    with trange(epochs) as t:

        for epoch in t:
            for batch_data, batch_target in dataset.data: # TODO: in line with 3D Molecular graph structure
                optimizer.zero_grad()

                # Run forward pass
                outputs = model(batch_data)

                # Compute loss
                (loss_name, loss), xtra_losses = loss_function(outputs, batch_target)
                loss.backward()
                optimizer.step()

            # Store train losses
            if epoch % store_train_every == 0:
                train_lss[epoch] = loss

                writer.add_scalar(f'TRAIN/{loss_name}', loss, epoch)
                if xtra_losses != None:
                    for name, loss_ in xtra_losses:
                        writer.add_scalar(f'TRAIN/{name}', loss_, epoch)

                # Print status
                t.set_description_str(f'Training Loss: {train_lss[epoch]:.3f} | Training MSE: {train_acc[epoch]:.3f} | Progress')

            if epoch % val_every_step == 0:
                model.eval()
                dataset.plot_regression_line(model, device, plot_uncertainty=True)
                model.train()

    # Close tensorboard
    writer.close()


if __name__ == '__main__':

    from Datasets import ToyDataset1D

    # Utilize GPU?
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load data
    toydata = ToyDataset1D(B=64, N=1024, device=device, visualize_on_load=True, seed=42)

    # Setup regression modelwork
    model = EvidentialToyModel1D(hidden_dim=32, device=device)
    model.to(device)

    # Define training parameters
    loss_function = NIGLoss(lambd_=1.0)
    optimizer = torch.optim.Adam(model.parameters())

    # Run training loop
    train(toydata, model, optimizer, loss_function=loss_function, lr=0.05,
          epochs=500, store_train_every=1, val_every_step=1,
          experiment_name='toy_example_1D', tensorboard_logdir='../logs')

