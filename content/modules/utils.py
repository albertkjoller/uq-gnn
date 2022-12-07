
import torch
import numpy as np


from content.modules.Datasets import ToyDataset1D, QM7_dataset, synthetic_dataset, get_loaders
from content.modules.GNNModels import EvidentialToyModel1D, EvidentialGNN3D, BaselineToyModel1D, BaselineGNN3D, ABaselineGNN3D, A2BaselineGNN3D
from content.modules.Losses import RMSELoss, NIGLoss, GAUSSIANNLLLoss

def retrieve_dataset(args):
    """
    Specifies which method to use for constructing the dataset.
    """

    device = torch.device(args.device)
    if args.dataset == 'TOY1D':
        return {'train': ToyDataset1D(B=args.batch_size, N=args.N_points, range_=(-4, 4), noise_level=args.toy_noise_level, device=device),
                'val': ToyDataset1D(B=args.batch_size, N=args.N_points, range_=(-4, 4), noise_level=args.toy_noise_level, device=device)}
    elif args.dataset == 'TOY1D-OOD':
        return {'test': ToyDataset1D(B=args.batch_size, N=args.N_points, range_=(-6, 6), OOD_boundaries=(-4, 4), noise_level=args.toy_noise_level, device=device)}

    elif args.dataset == 'QM7':
        return QM7_dataset(path=f"{args.data_dir}/QM7/qm7.mat", device=device)
    elif 'SYNTHETIC' in args.dataset:
        return synthetic_dataset(path = f"{args.data_dir}/{args.dataset}/", device=device)

def load_data(args):
    """
    Creates or loads pre-created dataloader objects for the dataset specified in input arguments.

    Parameters
    ----------
    args: input arguments given in the run file.

    Returns
    -------
    Dict of dataloaders (pytorch classes) for the training-, test- and validation splits of the dataset.
    """
    try:
        loaders = {dset_type: torch.load(f"{args.data_dir}/{args.dataset}/{dset_type}_loader_BZ{args.batch_size}.pth") for dset_type in ['train', 'val', 'test']}
        print("\nLOADED DATALOADER OBJECTS!")
    except FileNotFoundError:
        # Load data
        dataset = retrieve_dataset(args)

        if args.dataset == 'TOY1D': # TODO: add test as ARON said?
            loaders = {'train': dataset['train'].batches, 'val': dataset['val'], 'test': dataset['val'].batches, 'visualization': dataset['train']}
        elif args.dataset == 'TOY1D-OOD':
            loaders = {'train': None, 'val': None, 'test': dataset['test'].batches}
        else:
            print("\nCREATING DATALOADER OBJECTS...")
            # Construct loaders
            graph_info, graph_data = dataset
            # you can adjust the test and val size here
            loaders = get_loaders(graph_info, graph_data, batch_size=args.batch_size, test_size=0.2, val_size=0.2,
                                  device=torch.device(args.device), random_state=args.seed, shuffle=True)

            # Save loaders
            for dset_type, loader in loaders.items():
                torch.save(loader, f'{args.data_dir}/{args.dataset}/{dset_type}_loader_BZ{args.batch_size}.pth')

            print("\nSUCCESSFULLY CREATED AND SAVED DATALOADER OBJECTS!")

    return loaders

def get_model_specifications(args):
    likelihood = None

    # MODEL
    if args.model == 'TOY1D':
        model = EvidentialToyModel1D()
    elif args.model == 'BASE1D':
        model = BaselineToyModel1D()
    elif args.model == 'BASE3D':
        model = BaselineGNN3D(device=torch.device(args.device))
    elif args.model == 'GNN3D':
        model = EvidentialGNN3D(device=torch.device(args.device))
    elif args.model == 'DEBUG3D':
        model = ABaselineGNN3D(device=torch.device(args.device))
    elif args.model == 'DEBUG3D2':
        model = A2BaselineGNN3D(device=torch.device(args.device))
    else:
        raise NotImplementedError("Specified model is currently not implemented.")

    if args.mode == 'train':
        # LOSS
        if args.loss_function == 'RMSE':
            loss_function = RMSELoss()
        elif args.loss_function == 'GAUSSIANNLL':
            loss_function = GAUSSIANNLLLoss()
        elif args.loss_function == 'NIG':
            assert args.NIG_lambda != None, "Specify NIG_lambda for using the NIG loss function..."
            loss_function = NIGLoss(lambd_=args.NIG_lambda)
        else:
            raise NotImplementedError("Specified loss function is currently not implemented.")

        # OPTIMIZER
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        return model.to(torch.device(args.device)), loss_function, optimizer
    else:
        return model.to(torch.device('cpu')), None, None


def save_model(model, args):
    torch.save(model.state_dict(), f"{args.save_path}/{args.experiment_name}/final.pth")

def load_model(args):
    model, _, _ = get_model_specifications(args)
    state_dict = torch.load(f"{args.save_path}/{args.experiment_name}/best.ckpt")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Function found here: https://stackoverflow.com/questions/52756152/tensorboard-extract-scalar-by-a-script
from tensorboard.backend.event_processing import event_accumulator

def _load_tb_run(path):
    event_acc = event_accumulator.EventAccumulator(path)
    event_acc.Reload()
    data = {}

    for tag in sorted(event_acc.Tags()["scalars"]):
        x, y = [], []

        for scalar_event in event_acc.Scalars(tag):
            x.append(scalar_event.step)
            y.append(scalar_event.value)

        data[tag] = (np.asarray(x), np.asarray(y))
    return data