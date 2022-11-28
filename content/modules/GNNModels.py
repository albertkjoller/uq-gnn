"""Simple invariant and equivariant graph neural networks."""
import torch

class GNNInvariant(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

    Keyword Arguments
    -----------------
        output_dim : Dimension of output (default 2)
        state_dim : Dimension of the node states (default 10)
        num_message_passing_rounds : Number of message passing rounds
            (default 3)
    """

    def __init__(self, output_dim=2, state_dim=10, num_message_passing_rounds=3, device='cpu'):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.state_dim = state_dim
        self.edge_dim = 1
        self.output_dim = output_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        '''# Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim+self.edge_dim, self.state_dim),
            torch.nn.Tanh())'''
        # big-ass message net:
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + 4, self.state_dim),
            torch.nn.Tanh())

        # State output network
        self.output_net = torch.nn.Linear(
            self.state_dim, self.output_dim)

        # Utilize GPU?
        self.device = device
        self.to(device)

    def forward(self, x):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x: GraphDataset
            A data set object that contains the following member variables:

            node_coordinates : torch.tensor (N x 2)
                2d coordinates for each of the N nodes in all graphs
            node_graph_index : torch.tensor (N)
                Index of which graph each node belongs to
            edge_list : torch.tensor (E x 2)
                Edges (to-node, from-node) in all graphs
            node_to, node_from : Shorthand for column 0 and 1 in edge_list
            edge_lengths : torch.tensor (E)
                Edge features

        Returns
        -------
        out : N x output_dim
            Neural network output

        """
        # Initialize node features to zeros
        self.state = torch.zeros([x.num_nodes, self.state_dim]).to(self.device)

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            coord_sum = torch.sum(torch.abs(x.node_coordinates[x.node_from]), axis=1)
            dot_prod_1 = dot3D(x.node_coordinates[x.node_from], x.node_coordinates[x.node_to]).view(x.edge_lengths.shape)
            # DOT: node_coordinates_to/from and edge directions -> 71%
            dot_prod_2 = dot3D(x.node_coordinates[x.node_from], x.edge_vectors).view(x.edge_lengths.shape)
            # Big message network
            inp = torch.cat((self.state[x.node_from], x.edge_lengths, coord_sum.view(x.edge_lengths.shape), dot_prod_1.view(x.edge_lengths.shape),
                                 dot_prod_2.view(x.edge_lengths.shape)), 1)
            message = self.message_net(inp)
            # dot of messages:
            '''# Input to message passing networks
            inp = torch.cat((self.state[x.node_from], x.edge_lengths), 1)

            # Message networks
            message = self.message_net(inp)'''

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).to(self.device)
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Output
        out = self.output_net(self.graph_state)
        return out



class EvidentialGNN3D(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

    Keyword Arguments
    -----------------
        output_dim : Dimension of output (default 2)
        state_dim : Dimension of the node states (default 10)
        num_message_passing_rounds : Number of message passing rounds
            (default 3)
    """

    def __init__(self, device, state_dim=10, num_message_passing_rounds=3):
        super().__init__()

        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 4

        # Message passing networks
        '''self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim+self.edge_dim, self.state_dim),
            torch.nn.Tanh())'''
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + 4, self.state_dim),
            torch.nn.Tanh())

        # Define layers
        layer1 = torch.nn.Linear(self.state_dim, self.output_dim)
        torch.nn.init.xavier_normal_(layer1.weight)

        # State output network --> (gamma, v, alpha, beta)
        self.output_net = torch.nn.Sequential(
            layer1,
        )

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

        # Utilize GPU? #TODO: move this out of the class itself and into the run-script
        self.device = device
        self.to(device)

    def forward(self, x):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x: GraphDataset
            A data set object that contains the following member variables:

            node_coordinates : torch.tensor (N x 2)
                2d coordinates for each of the N nodes in all graphs
            node_graph_index : torch.tensor (N)
                Index of which graph each node belongs to
            edge_list : torch.tensor (E x 2)
                Edges (to-node, from-node) in all graphs
            node_to, node_from : Shorthand for column 0 and 1 in edge_list
            edge_lengths : torch.tensor (E)
                Edge features

        Returns
        -------
        out : N x output_dim
            Neural network output

        """
        # Initialize node features to zeros
        self.state = torch.zeros([x.num_nodes, self.state_dim]).to(self.device)

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            coord_sum = torch.sum(torch.abs(x.node_coordinates[x.node_from]), axis=1)
            dot_prod_1 = dot3D(x.node_coordinates[x.node_from], x.node_coordinates[x.node_to]).view(x.edge_lengths.shape)
            # DOT: node_coordinates_to/from and edge directions -> 71%
            dot_prod_2 = dot3D(x.node_coordinates[x.node_from], x.edge_vectors).view(x.edge_lengths.shape)
            # Big message network
            inp = torch.cat((self.state[x.node_from], x.edge_lengths.to(self.device), coord_sum.view(x.edge_lengths.shape).to(self.device),
                             dot_prod_1.view(x.edge_lengths.shape),#), 1)
                             dot_prod_2.view(x.edge_lengths.shape)), 1)
            message = self.message_net(inp)
            '''# Input to message passing networks
            inp = torch.cat((self.state[x.node_from], x.edge_lengths), 1)

            # Message networks
            message = self.message_net(inp)'''

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).to(self.device)
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.output_net(self.graph_state) # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        gamma, v, alpha, beta = torch.tensor_split(evidential_params_, 4, axis=1)
        out = torch.concat([gamma, self.softplus(v), self.softplus(alpha) + 1, self.softplus(beta)], axis=1)
        return out



class BaselineToyModel1D(torch.nn.Module):
    """
    Toy baseline model for 3rd order polynomial.
    """

    def __init__(self, hidden_dim=100):
        super().__init__()

        # Regression network for 1D toy task
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1), # (mu)
        )
        # Initialize weights
        self.net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

    def init_weights(self, layer): # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.data
        # Get parameters of NIG distribution (4-dimensional output)
        mu = self.net(x) # (mu)
        return mu

class EvidentialToyModel1D(torch.nn.Module):
    """
    Toy model for verifying that evidential learning works for approximating a 3rd order polynomial.
    """

    def __init__(self, hidden_dim=100, eps=1e-7):
        super().__init__()

        # Regression network for 1D toy task
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 4), # (gamma, nu, alpha, beta)
        )
        # Initialize weights
        self.net.apply(self.init_weights)
        self.eps = eps

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

    def init_weights(self, layer): # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.data
        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.net(x) # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        gamma, nu, alpha, beta = torch.tensor_split(evidential_params_, 4, axis=1)
        out = torch.concat([gamma, self.softplus(nu) + self.eps, self.softplus(alpha).to(torch.float64).add(1) + self.eps, self.softplus(beta) + self.eps], axis=1)
        return out


### INVARIANT OPERATIONS

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

