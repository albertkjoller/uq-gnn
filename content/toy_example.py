
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

from GNNModels import EvidentialToyModel
from losses import NIGLoss

def train(train_data, model, optimizer, lr=0.01, loss_function=torch.nn.MSELoss(), lambd=1.0,
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
            for batch_data, batch_target in train_data:
                optimizer.zero_grad()
                # Run forward pass
                outputs = model(batch_data)

                # Compute loss
                loss, (nll_loss, reg_loss) = loss_function(outputs).compute_loss(batch_target, lambd)
                loss.backward()
                optimizer.step()

            # Store train losses
            if epoch % store_train_every == 0:
                writer.add_scalar('TRAIN/Loss', loss, epoch)
                writer.add_scalar('TRAIN/NLL Loss', nll_loss, epoch)
                writer.add_scalar('TRAIN/REG Loss', reg_loss, epoch)

                train_lss[epoch] = loss
                # TODO: add MSE for batch
                #train_acc[epoch] = torch.mean(torch.pow(outputs[:, 0] - train_data['target'], 2))

                # Print status
                t.set_description_str(f'Training Loss: {train_lss[epoch]:.3f} | Training MSE: {train_acc[epoch]:.3f} | Progress')
                #print("")

            if epoch % val_every_step == 0:
                model.eval()
                # TODO: input data (black dots) are currently not used
                plot_regression_line(model, device, plot_uncertainty=True)
                model.train()
    # Close tensorboard
    writer.close()

def plot_regression_line(model, device, plot_uncertainty=False):
    # Predict on the data range
    toy_ = torch.arange(-6, 6, 12 / N).reshape(-1, 1).to(device)
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
        plt.fill_between(results['xaxis'], results['y_pred'] - results['epistemic'], results['y_pred'] + results['epistemic'], alpha=.3, interpolate=True)  # step='post')

    plt.xlim([-6.5, 6.5])
    plt.ylim([-6.5**3, 6.5**3])
    plt.show()

if __name__ == '__main__':

    # Utilize GPU?
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define data
    N = 1024
    B = 64

    data = {}
    order_ = torch.randperm(N) # shuffle data
    data['data'] = torch.arange(-4, 4, 8/N)[order_].reshape(-1, B, 1).to(device)
    data['target'] = data['data']**3 + (torch.randn(N, 1).reshape(-1, B, 1) * 3).to(device)

    # plot data
    plt.plot(data['data'].detach().flatten().cpu(), data['target'].detach().flatten().cpu(), '.')
    plt.plot(torch.arange(-4, 4, 8/N), torch.arange(-4, 4, 8/N)**3, 'k--')
    plt.xlim([-6.5, 6.5])
    plt.show()

    # structure data in batches
    data = list(zip(data['data'], data['target']))

    # Setup regression modelwork
    model = EvidentialToyModel(hidden_dim=32, device=device)
    model.to(device)

    loss_function = NIGLoss
    optimizer = torch.optim.Adam(model.parameters(),)

    # Run training loop
    train(data, model, optimizer, loss_function=loss_function, lr=0.05,
          epochs=500, store_train_every=1, val_every_step=1, lambd=0.0, # TODO: investigate lambda
          experiment_name='toy_example_1D', tensorboard_logdir='../logs')


    print("")

