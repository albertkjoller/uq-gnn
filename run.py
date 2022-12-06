
import os
import torch
import numpy as np
import argparse
from copy import deepcopy

from content.train import train
from content.evaluate import evaluate_model
from content.modules.utils import load_data, get_model_specifications, save_model, load_model

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
    #parser.add_argument(
    #    "--dataset",
    #    type=str,
    #    help="Name of the dataset to be used. Options: [TOY1D, QM7, SYNTHETIC]",
    #    required=True,
    #)
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
    #parser.add_argument(
    #    "--model",
    #    type=str,
    #    help="The model type to use when training. Currently, either 'TOY1D', 'GNN3D', 'BASE1D', or 'BASE3D'."
    #)
    parser.add_argument(
        "--model",
        action='append',
        help="The model type to use when training. Currently, either 'TOY1D', 'GNN3D', 'BASE1D', or 'BASE3D'."
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        help="Type of loss function. Either NIG or MSE or GAUSSIANNLLoss.",
        default='NIG',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="How many epochs to run (in 'train'-mode).",
        default=500,
    )
    parser.add_argument(
        "--NIG_lambda",
        type=float,
        help="Lambda value when running NIG loss function.",
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
    #parser.add_argument(
    #    "--experiment_name",
    #    type=str,
    #    help="Name of experiment run (identical for train and evaluation mode).",
    #    required=True,
    #)
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
    # TODO: add more assertions for input arguments

def determine_run_version(args):

    version_ = args.experiment_name
    version_ += f"_data.{args.dataset}"
    version_ += f"_loss.{args.loss_function}"
    if args.loss_function == 'NIG':
        version_ += f"_lambda.{args.NIG_lambda}"
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

    # TRAINING MODE
    if args.mode == 'train':
        # currently in list (because needed in evaluation)
        args.dataset = args.dataset[0]
        args.experiment_name = args.experiment_name[0]
        args.model = args.model[0]

        try:
            os.makedirs(args.save_path + f"/{args.experiment_name}")
        except Exception as e:
            print('Include --save_path, or folder already exists')
        # Load data and device
        # todo: this causes error: 'visualization': dataset['train']
        loaders = load_data(args)
        device = torch.device(args.device)


        # Get training specific objects
        model, loss_function, optimizer = get_model_specifications(args)

        # Run training loop
        model, best_epoch = train(loaders, model, optimizer,
                                  loss_function=loss_function,
                                  epochs=args.epochs,
                                  val_every_step=args.val_every_step,
                                  experiment_name=args.experiment_name,
                                  tensorboard_logdir=args.tensorboard_logdir,
                                  tensorboard_filename=determine_run_version(args),
                                  save_path=f"{args.save_path}/{args.experiment_name}",
                                  )

        if args.save_path != '':
            save_model(model, args)

    # todo: run and evaluate argument?
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

        evaluate_model(loaders_dict=loaders_dict, models=models, experiments=args.experiment_name, args = args)


