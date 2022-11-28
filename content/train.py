import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import trange

import torch
from torch.utils.tensorboard import SummaryWriter

def train(dataloaders, model, optimizer, loss_function, epochs=1000,
          val_every_step=50, experiment_name=str(time.time()),
          tensorboard_logdir='logs', tensorboard_filename=str(time.time())):

    # Setup tensorboard
    writer = SummaryWriter(Path(f"{tensorboard_logdir}")/tensorboard_filename)

    # unpacking loaders
    train_loader, val_loader, test_loader = tuple(dataloaders.values())

    # Setup parameters
    model.train()

    # Setup storage information
    train_lss, val_lss = np.zeros(epochs), np.zeros(epochs // val_every_step)

    with trange(epochs) as t:

        val_step = -1
        for epoch in t:
            batch_loss, batch_xtra_losses = [], defaultdict(list)
            for idx_batch, train_batch in enumerate(train_loader):

                optimizer.zero_grad()
                # Run forward pass
                outputs = model(train_batch)

                # Compute loss
                (loss_name, loss), xtra_losses = loss_function(outputs, train_batch.target)
                loss.backward()
                optimizer.step()

                # Store batch loss for epoch
                batch_loss.append(loss.item())
                if loss_function.__class__.__name__ == "NIGLoss":
                    for name, loss_ in xtra_losses.items():
                        batch_xtra_losses[name] += [loss_.item()]

            # Store training losses
            train_lss[epoch] = np.mean(batch_loss)
            writer.add_scalar(f'TRAIN/{loss_name}', np.mean(batch_loss), epoch)
            if loss_function.__class__.__name__ == "NIGLoss":
                for name, loss_ in batch_xtra_losses.items():
                    writer.add_scalar(f'TRAIN/{name}', np.mean(loss_), epoch)

            # Validation run
            if epoch % val_every_step == 0:
                with torch.no_grad():
                    # Change network mode
                    model.eval()
                    # Evaluate on validation batch
                    val_loss = evaluate(model, val_loader, writer, epoch, loss_function, experiment_name)
                    val_lss[val_step] = np.mean(val_loss)
                    # Switch back to training mode
                    model.train()

                val_step += 1

            # Print status
            t.set_description_str(f'Training Loss: {train_lss[epoch-1]:.3f} \t| \t Validation Loss: {val_lss[val_step-1]:.3f} | Progress')

    # Close tensorboard
    writer.close()

def evaluate(model, data, writer, epoch, loss_function, experiment_name):

    if '1D' in model.__class__.__name__:
        save_path = Path(f"results/{experiment_name}")
        if 'Evidential' in model.__class__.__name__:
            os.makedirs(save_path / 'epistemic', exist_ok=True)
            os.makedirs(save_path / 'aleatoric', exist_ok=True)

            data.plot_regression_line(model, epoch=epoch, uncertainty_type='epistemic', save_path=save_path / 'epistemic', show=True)
            data.plot_regression_line(model, epoch=epoch, uncertainty_type='aleatoric', save_path=save_path / 'aleatoric', show=True)
        else:
            os.makedirs(save_path / 'baseline', exist_ok=True)
            data.plot_regression_line(model, epoch, uncertainty_type='baseline', save_path=save_path / 'baseline', show=False)

        loss = 0

    elif model.__class__.__name__ == 'EvidentialGNN3D':

        batch_loss, batch_xtra_losses = [], defaultdict(list)
        for idx_batch, batch in enumerate(data):
            # Compute return of model
            outputs = model(batch)

            # Compute loss
            (loss_name, loss), xtra_losses = loss_function(outputs, batch.target)

            batch_loss.append(loss.item())
            if loss_function.__class__.__name__ == "NIGLoss":
                for name, loss_ in xtra_losses.items():
                    batch_xtra_losses[name] += [loss_.item()]

        loss = np.mean(batch_loss)
        writer.add_scalar(f'VALIDATION/{loss_name}', loss, epoch)
        if loss_function.__class__.__name__ == "NIGLoss":
            for name, loss_ in batch_xtra_losses.items():
                writer.add_scalar(f'VALIDATION/{name}', np.mean(loss_), epoch)

    return loss