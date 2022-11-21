"""Simple invariant and equivariant graph neural networks."""
import torch
from utils import cross2D, dot2D, dot3D

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
            dot_prod_1 = dot(x.node_coordinates[x.node_from], x.node_coordinates[x.node_to]).view(x.edge_lengths.shape)
            # DOT: node_coordinates_to/from and edge directions -> 71%
            dot_prod_2 = dot(x.node_coordinates[x.node_from], x.edge_vectors).view(x.edge_lengths.shape)
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

    def __init__(self, state_dim=10, num_message_passing_rounds=3, device='cpu'):
        super().__init__()

        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 4

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim+self.edge_dim, self.state_dim),
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
            # Input to message passing networks
            inp = torch.cat((self.state[x.node_from], x.edge_lengths), 1)

            # Message networks
            message = self.message_net(inp)

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


class EvidentialToyModel1D(torch.nn.Module):
    """
    Toy model for verifying that evidential learning works for approximating a 3rd order polynomial.
    """

    def __init__(self, hidden_dim=8, device='cpu'):
        super().__init__()

        # Regression network for 1D toy task
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 4), # (gamma, v, alpha, beta)
        )

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.net(x) # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        gamma, v, alpha, beta = torch.tensor_split(evidential_params_, 4, axis=1)
        out = torch.concat([gamma, self.softplus(v), self.softplus(alpha) + 1, self.softplus(beta)], axis=1)
        return out