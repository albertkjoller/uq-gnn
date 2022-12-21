
import os
import torch
import numpy as np
import argparse
from copy import deepcopy

from content.train import train
from content.evaluate import evaluate_model
from content.modules.utils import load_data, get_model_specifications, save_model, load_model, get_scalar

def get_arguments(parser):
    parser.add_argument(
        "--mode",
        type=str,
        help="Defines whether to run in train or evaluation mode.",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for pseudo-random behaviour.",
        default=42,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory where various datasets are stored.",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        action='append',
        help="Name of the dataset to be used. Options: [TOY1D, QM7, SYNTHETIC]"
    )
    parser.add_argument(
        "--toy_noise_level",
        type=float,
        help="Noise level when generating the data.",
        default=3.0,
    )
    parser.add_argument(
        "--N_points",
        type=int,
        help="Number of points for TOY1D dataset.",
        default=1024
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to be used.",
        default=64,
    )
    parser.add_argument(
        "--model",
        action='append',
        help="The model type to use when training. Currently, either 'TOY1D', 'GNN3D', 'BASE1D', or 'BASE3D'."
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        help="Type of loss function. Either NIG or MSE or GAUSSIANNLL.",
        default='NIG',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="How many epochs to run (in 'train'-mode).",
        default=500,
    )
    parser.add_argument(
        "--scalar",
        type=str,
        help="Type of scalar on target variable.",
        default=None,
        choices=['standardize'],
    )
    parser.add_argument(
        "--NIG_lambda",
        nargs='+',
        action='append',
        help="Lambda value when running NIG loss function.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        help="Trade-off between specific loss and RMSE. Must be in range [0, 1]. A value of 1 means full emphasis on the RMSE.",
        default=0,
    )
    parser.add_argument(
        "--kappa_decay",
        type=float,
        help="Decay parameter for threshold value between loss and RMSE. Must be in range [0, 1] with 1 being no decay.",
        default=1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate when training.",
    )
    parser.add_argument(
        "--val_every_step",
        type=int,
        help="Frequency of running model on validation data during training.",
        default=25,
    )
    parser.add_argument(
        "--tensorboard_logdir",
        type=str,
        help="Path to location for storing tensorboard log-files.",
        default='logs',
    )
    parser.add_argument(
        '--experiment_name',
        action='append',
        help="Name of experiment run (identical for train and evaluation mode).",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run computations on. Either 'cpu' or 'cuda'.",
        default='cpu',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to saving models.",
        default=''
    )

    parser.add_argument(
        '--id_ood',
        action='append',
        help="For evaluation only: whether dataset is in or out of distribution.",
    )

def check_assertions(args):
    # Mode
    assert args.mode in ['train', 'evaluation'], "run.py only support 'train' or 'evaluation' mode"
    # Device
    if args.device == 'cuda':
        assert torch.cuda.is_available() == True, "Your system does not support running on GPU."

def determine_run_version(args):
    version_ = args.experiment_name
    version_ += f"_data.{args.dataset}"
    version_ += f"_loss.{args.loss_function}"
    if args.loss_function == 'NIG':
        version_ += f"_lambda.{args.NIG_lambda}"
    version_ += f"_kappa.{args.kappa}"
    version_ += f"_decay.{args.kappa_decay}"
    version_ += f"_epochs.{args.epochs}"
    version_ += f"_lr.{args.lr}"
    version_ += f"_batch.{args.batch_size}"
    version_ += f"_seed.{args.seed}"

    return version_

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description="Runs associated methods for Invariant Graph Neural Networks.")
    get_arguments(parser)
    args = parser.parse_args()

    # Check if inconsistence within input arguments
    check_assertions(args)
    # setting seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TRAINING MODE
    if args.mode == 'train':
        # currently in list (because needed in evaluation)
        args.dataset = args.dataset[0]
        args.experiment_name = args.experiment_name[0]
        args.model = args.model[0]

        # Load data and device
        # todo: this causes error: 'visualization': dataset['train']
        loaders = load_data(args)
        loaders = {k: v for k, v in loaders.items() if k != 'visualization'}
        device = torch.device(args.device)

        # Get training specific objects
        model, loss_function, optimizer = get_model_specifications(args)

        # if target is scaled
        if args.scalar is not None:
            scalar = get_scalar(loaders['train'], args.scalar)
            model.scalar = scalar
            loss_function.scalar = scalar

        # Grid search Lambda values
        lambda_vals = args.NIG_lambda[0] # extracting values before overwrite
        for lambda_val in lambda_vals:
            args.NIG_lambda = float(lambda_val)

            try: # Creating folder
                os.makedirs(args.save_path + f"/{args.experiment_name}" + f"/lambda{args.NIG_lambda}")
            except Exception as e:
                print('Include --save_path, or folder already exists')
                
            # Get training specific objects
            model, loss_function, optimizer = get_model_specifications(args)

            # Run training loop
            model, best_epoch = train(loaders, model, optimizer,
                                    loss_function=loss_function,
                                    epochs=args.epochs,
                                    kappa=args.kappa,
                                    kappa_decay=args.kappa_decay,
                                    val_every_step=args.val_every_step,
                                    experiment_name=args.experiment_name,
                                    tensorboard_logdir=args.tensorboard_logdir,
                                    tensorboard_filename=determine_run_version(args),
                                    save_path=f"{args.save_path}/{args.experiment_name}",
                                    )

            if args.save_path != '':
                save_model(model, args)

    elif args.mode == 'evaluation':
        models = {}
        loaders_dict = {}
        # getting each model
        for idx, exp in enumerate(args.experiment_name):
            curr_args = deepcopy(args)
            curr_args.experiment_name, curr_args.model = exp, args.model[idx]
            # model
            models[exp] = load_model(curr_args)

        # getting each dataset loader
        for idx, data in enumerate(args.dataset):
            curr_args = deepcopy(args)
            curr_args.dataset =args.dataset[idx]
            # dataset
            loaders_dict[args.id_ood[idx]] = load_data(curr_args)

        results_ = evaluate_model(loaders_dict=loaders_dict, models=models, experiments=args.experiment_name, args = args)
        print(results_)



