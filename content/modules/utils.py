
import torch

from content.modules.Datasets import ToyDataset1D, QM7_dataset, synthetic_dataset, get_loaders
from content.modules.GNNModels import EvidentialToyModel1D, EvidentialGNN3D, BaselineToyModel1D, BaselineGNN3D
from content.modules.Losses import RMSELoss, NIGLoss, GAUSSIANNLLLoss

def retrieve_dataset(args):
    """
    Specifies which method to use for constructing the dataset.
    """

    device = torch.device(args.device)
    if args.dataset == 'TOY1D':
        return {'train': ToyDataset1D(B=args.batch_size, N=args.N_points, range_=(-4, 4), noise_level=args.toy_noise_level, device=device),
                'val': ToyDataset1D(B=args.batch_size, N=args.N_points, range_=(-6, 6), noise_level=args.toy_noise_level, device=device)}

    elif args.dataset == 'QM7':
        return QM7_dataset(path=f"{args.data_dir}/QM7/qm7.mat", device=device)
    elif 'SYNTHETIC' in args.dataset:
        return synthetic_dataset(path = f"{args.data_dir}/{args.dataset}/", device=device)
        #raise NotImplementedError("TODO: fix path to dataset - what is this?")
        #return synthetic_dataset(f"{args.data_dir}/SYNTHETIC/...") # TODO: ME!

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

        if args.dataset == 'TOY1D':
            loaders = {'train': dataset['train'].batches, 'val': dataset['val'], 'visualization': dataset['train']}
        else:
            print("\nCREATING DATALOADER OBJECTS...")
            # Construct loaders
            graph_info, graph_data = dataset
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