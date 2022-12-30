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
        self.model_type = 'evidential'

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


class BaselineGNN3D(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

        Keyword Arguments
        -----------------
            output_dim : Dimension of output (default 2)
            state_dim : Dimension of the node states (default 10)
            num_message_passing_rounds : Number of message passing rounds
                (default 3)
        """

    def __init__(self, device, state_dim=20, num_message_passing_rounds=3):
        super().__init__()
        self.model_type = 'baseline'

        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 2
        self.num_features = 4
        self.hidden_dim = 40

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.state_dim + self.num_features).double(),
            torch.nn.Linear(self.state_dim + self.num_features, self.hidden_dim).double(),
            torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, 150).double(),
            torch.nn.Dropout(0.2),
            torch.nn.Tanh(),
            torch.nn.Linear(150, self.hidden_dim).double(),
            torch.nn.Dropout(0.8),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, self.state_dim).double(),
            torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
        )

        # Output net
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.output_dim).double()  # (gamma, v, alpha, beta)
        )

        # Initialize weights
        self.message_net.apply(self.init_weights)
        self.output_net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

        # Utilize GPU? #TODO: move this out of the class itself and into the run-script
        self.device = device
        self.to(device)

    def init_weights(self,
                     layer):  # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                          nonlinearity='relu').double()  # Kaiming for Relu

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
        self.state = torch.zeros([x.num_nodes, self.state_dim]).double().to(self.device)

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            e_len = x.edge_lengths
            coord_difference = torch.sum(torch.abs(x.node_coordinates[x.node_from] - x.node_coordinates[x.node_to]),
                                         axis=1)
            dot_coord = dot3D(x.node_coordinates[x.node_from], x.node_coordinates[x.node_to]).view(x.edge_lengths.shape)
            dot_diff = dot3D((x.node_coordinates[x.node_from] - x.node_coordinates[x.node_to]), x.edge_vectors).view(
                x.edge_lengths.shape)

            # Stacking features
            inp = torch.cat((self.state[x.node_from],
                             e_len,
                             coord_difference.view(e_len.shape),
                             dot_coord.view(e_len.shape),
                             dot_diff.view(e_len.shape),
                             ), 1)
            message = self.message_net(inp)

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).double().to(self.device)
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.output_net(self.graph_state)  # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        mu, sigma = torch.tensor_split(evidential_params_, 2, axis=1)
        out = torch.concat([mu, self.softplus(sigma)], axis=1)
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

    def __init__(self, device, state_dim=20, num_message_passing_rounds=3, eps=1e-5):
        super().__init__()
        self.model_type = 'evidential'

        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 4
        self.num_features = 4
        self.hidden_dim = 40

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.state_dim + self.num_features).double(),
            torch.nn.Linear(self.state_dim + self.num_features, self.hidden_dim).double(),
            torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, 150).double(),
            torch.nn.Dropout(0.2),
            torch.nn.Tanh(),
            torch.nn.Linear(150, self.hidden_dim).double(),
            torch.nn.Dropout(0.8),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, self.state_dim).double(),
            torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
        )

        # Output net
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.output_dim).double() # (gamma, v, alpha, beta)
        )

        # Initialize weights
        self.message_net.apply(self.init_weights)
        self.output_net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

        # Utilize GPU? #TODO: move this out of the class itself and into the run-script
        self.device = device
        self.eps = eps
        self.to(device)

    def init_weights(self, layer): # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu').double() # Kaiming for Relu


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
        self.state = torch.zeros([x.num_nodes, self.state_dim]).to(self.device).double()

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            # Preparing features
            e_len = x.edge_lengths
            coord_difference = torch.sum(torch.abs(x.node_coordinates[x.node_from] - x.node_coordinates[x.node_to]), axis=1)
            dot_coord = dot3D(x.node_coordinates[x.node_from], x.node_coordinates[x.node_to]).view(x.edge_lengths.shape)
            dot_diff = dot3D((x.node_coordinates[x.node_from] - x.node_coordinates[x.node_to]), x.edge_vectors).view(x.edge_lengths.shape)

            # Stacking features
            inp = torch.cat((self.state[x.node_from],
                            e_len,
                            coord_difference.view(e_len.shape),
                            dot_coord.view(e_len.shape),
                            dot_diff.view(e_len.shape),
                            ), 1)

            message = self.message_net(inp)

            if int(sum(sum(torch.isnan(message)))) > 0:
                stop = 1

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)
        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).to(self.device).double()
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.output_net(self.graph_state) # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        gamma, v, alpha, beta = torch.tensor_split(evidential_params_, 4, axis=1)
        out = torch.concat([gamma, self.softplus(v) + self.eps, self.softplus(alpha).add(1.0).to(torch.float64)+self.eps, self.softplus(beta) + self.eps], axis=1)
        return out

class BaselineToyModel1D(torch.nn.Module):
    """
    Toy baseline model for 3rd order polynomial.
    """

    def __init__(self, hidden_dim=100):
        super().__init__()
        self.model_type = 'baseline'

        # Regression network for 1D toy task
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2), # (mu, sigma)
        )
        # Initialize weights
        self.net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

    def init_weights(self,layer):  # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.data
        # Get parameters of NIG distribution (4-dimensional output)
        gaussian_params_ = self.net(x)  # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        mu, sigma = torch.tensor_split(gaussian_params_, 2, axis=1)
        #sigma = sigma.reshape(-1,1)
        out = torch.concat([mu, self.softplus(sigma)], axis=1)
        return out

class EvidentialToyModel1D(torch.nn.Module):
    """
    Toy model for verifying that evidential learning works for approximating a 3rd order polynomial.
    """

    def __init__(self, hidden_dim=100, eps=1e-7):
        super().__init__()
        self.model_type = 'evidential'

        # Regression network for 1D toy task
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
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




############ testing ############


class Baseline_Q7_test(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

        Keyword Arguments
        -----------------
            output_dim : Dimension of output (default 2)
            state_dim : Dimension of the node states (default 10)
            num_message_passing_rounds : Number of message passing rounds
                (default 3)
        """

    def __init__(self, device, state_dim=32, num_message_passing_rounds=5):
        super().__init__()
        self.model_type = 'baseline'
        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 2
        self.num_features = 1
        self.hidden_dim_message = 128
        self.hidden_dim_output = 128

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.num_features, self.hidden_dim_message).double(),
            torch.nn.LayerNorm(self.hidden_dim_message).double(),
            #torch.nn.BatchNorm1d(self.hidden_dim_message).double(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_dim_message, self.state_dim).double(),
            torch.nn.LayerNorm(self.state_dim).double(),
            #torch.nn.BatchNorm1d(self.state_dim).double(),
            torch.nn.LeakyReLU(),
        )

        # Output net
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim_output).double(),
            # --
            # standardize then include:
            torch.nn.LayerNorm(self.hidden_dim_output).double(),
            torch.nn.LeakyReLU(),
            # --
            torch.nn.Linear(self.hidden_dim_output, self.output_dim).double(),
        )

        # Initialize weights
        self.message_net.apply(self.init_weights)
        self.output_net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

        # Utilize GPU? #TODO: move this out of the class itself and into the run-script
        self.device = device
        self.to(device)
        self.scalar = None


    def init_weights(self,
                     layer):  # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                          nonlinearity='leaky_relu').double()  # Kaiming for Relu


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
        self.state = torch.zeros([x.num_nodes, self.state_dim]).double().to(self.device)

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            e_len = x.edge_lengths
            e_coulomb = x.edge_coulomb

            # Stacking features
            inp = torch.cat((self.state[x.node_from],
                             e_coulomb,
                             ), 1)
            message = self.message_net(inp)

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).double().to(self.device)
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.output_net(self.graph_state)  # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        mu, var = torch.tensor_split(evidential_params_, 2, axis=1)
        var = self.softplus(var) # converting to positive
        # if trained on scaled and in eval mode
        if self.scalar is not None and self.training==False:
            # de-scaling prediction
            mu = torch.from_numpy(self.scalar.inverse_transform(mu.detach()))
            var = var.detach()*self.scalar.var_

        out = torch.concat([mu, var], axis=1)
        return out


class EvidentialQM7_3D(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

        Keyword Arguments
        -----------------
            output_dim : Dimension of output (default 2)
            state_dim : Dimension of the node states (default 10)
            num_message_passing_rounds : Number of message passing rounds
                (default 3)
        """

    def __init__(self, device, state_dim=32, num_message_passing_rounds=5, eps=1e-10):
        super().__init__()
        self.model_type = 'evidential'
        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 4
        self.num_features = 1
        self.h1_message = 128
        self.h2_message = 256
        self.h3_message = 256
        self.hidden_dim_output = 128

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.num_features, self.h1_message).double(),
            torch.nn.LayerNorm(self.h1_message).double(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.h1_message, self.state_dim).double(),
            torch.nn.LayerNorm(self.state_dim).double(),
            torch.nn.LeakyReLU(),
        )

        # Output net
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim_output).double(),
            torch.nn.Linear(self.hidden_dim_output, self.output_dim).double()
        )

        # Initialize weights
        self.message_net.apply(self.init_weights)
        self.output_net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

        # Utilize GPU? #TODO: move this out of the class itself and into the run-script
        self.eps = eps
        self.device = device
        self.to(device)

        self.scalar = None

    def init_weights(self,
                     layer):  # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                          nonlinearity='leaky_relu').double()  # Kaiming for Relu

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
        self.state = torch.zeros([x.num_nodes, self.state_dim]).double().to(self.device)

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            e_len = x.edge_lengths
            e_coulomb = x.edge_coulomb
            
            # Stacking features
            inp = torch.cat((self.state[x.node_from],
                             e_coulomb, 
                             ), 1)
            message = self.message_net(inp)

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).double().to(self.device)
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.output_net(self.graph_state)  # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        gamma, v, alpha, beta = torch.tensor_split(evidential_params_, 4, axis=1)

        # if trained on scaled and in eval mode
        if self.scalar is not None and self.training==False:
            # de-scaling prediction
            gamma = torch.from_numpy(self.scalar.inverse_transform(gamma.detach()))

        out = torch.concat(
            [gamma, self.softplus(v) + self.eps, self.softplus(alpha).add(1.0).to(torch.float64) + self.eps,
             self.softplus(beta) + self.eps], axis=1)

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




# TODO: remove




class ABaselineGNN3D(torch.nn.Module):
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
        self.model_type = 'baseline'

        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 2
        self.num_features = 3
        self.h1 = 48
        self.h2 = 96
        self.h3 = 48

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.num_features, self.h1),
            torch.nn.BatchNorm1d(self.h1),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),

            torch.nn.Linear(self.h1, self.h2),
            torch.nn.BatchNorm1d(self.h2),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),

            torch.nn.Linear(self.h2, self.h2),
            torch.nn.BatchNorm1d(self.h2),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),

            torch.nn.Linear(self.h2, self.state_dim),
        )

        # Output net
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.output_dim),  # (gamma, nu, alpha, beta)
        )

        # Initialize weights
        self.message_net.apply(self.init_weights)
        self.output_net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

        # Utilize GPU? #TODO: move this out of the class itself and into the run-script
        self.device = device
        self.to(device)

    def init_weights(self,
                     layer):  # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)

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

            # Preparing features
            e_len = x.edge_lengths.to(self.device)
            coord_difference = torch.sum(torch.abs(
                x.node_coordinates[x.node_from].to(self.device) - x.node_coordinates[x.node_to].to(self.device)),
                                         axis=1)
            # dot_coord = dot3D(x.node_coordinates[x.node_from].to(self.device), x.node_coordinates[x.node_to].to(self.device)).view(x.edge_lengths.shape)
            dot_diff = dot3D(
                (x.node_coordinates[x.node_from].to(self.device) - x.node_coordinates[x.node_to].to(self.device)),
                x.edge_vectors.to(self.device)).view(x.edge_lengths.shape)

            # Stacking features
            inp = torch.cat((self.state[x.node_from],
                             e_len,
                             coord_difference.view(e_len.shape),
                             dot_diff.view(e_len.shape),
                             ), 1)

            message = self.message_net(inp)

            if int(sum(sum(torch.isnan(message)))) > 0:
                stop = 1

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to.to(self.device), message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).to(self.device)
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.output_net(self.graph_state)  # (gamma, v, alpha, beta)

        # Apply activations as specified after Equation 10 in the paper
        mu, sigma = torch.tensor_split(evidential_params_, 2, axis=1)
        out = torch.concat([mu, self.softplus(sigma)], axis=1)
        return out



class A2BaselineGNN3D(torch.nn.Module):
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
        self.model_type = 'baseline'

        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 2
        self.num_features = 3
        self.h1 = 48
        self.h2 = 96
        self.h3 = 48

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.num_features, self.h1),
            torch.nn.BatchNorm1d(self.h1),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),

            torch.nn.Linear(self.h1, self.h2),
            torch.nn.BatchNorm1d(self.h2),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),

            torch.nn.Linear(self.h2, self.h2),
            torch.nn.BatchNorm1d(self.h2),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),

            torch.nn.Linear(self.h2, self.state_dim),
        )

        # Output net
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.output_dim),  # (gamma, nu, alpha, beta)
        )

        # Initialize weights
        self.message_net.apply(self.init_weights)
        self.output_net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

        # Utilize GPU? #TODO: move this out of the class itself and into the run-script
        self.device = device
        self.to(device)

    def init_weights(self,
                     layer):  # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)

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

            # Preparing features
            e_len = x.edge_lengths.to(self.device)
            coord_difference = torch.sum(torch.abs(
                x.node_coordinates[x.node_from].to(self.device) - x.node_coordinates[x.node_to].to(self.device)),
                                         axis=1)
            # dot_coord = dot3D(x.node_coordinates[x.node_from].to(self.device), x.node_coordinates[x.node_to].to(self.device)).view(x.edge_lengths.shape)
            dot_diff = dot3D(
                (x.node_coordinates[x.node_from].to(self.device) - x.node_coordinates[x.node_to].to(self.device)),
                x.edge_vectors.to(self.device)).view(x.edge_lengths.shape)

            # Stacking features
            inp = torch.cat((self.state[x.node_from],
                             e_len,
                             coord_difference.view(e_len.shape),
                             dot_diff.view(e_len.shape),
                             ), 1)

            message = self.message_net(inp)

            if int(sum(sum(torch.isnan(message)))) > 0:
                stop = 1

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to.to(self.device), message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).to(self.device)
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.output_net(self.graph_state)  # (gamma, v, alpha, beta)

        # Apply activations as specified after Equation 10 in the paper
        mu, sigma = torch.tensor_split(evidential_params_, 2, axis=1)
        out = torch.concat([mu, self.softplus(sigma)], axis=1)
        return out



class EvidentialQM7(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

        Keyword Arguments
        -----------------
            output_dim : Dimension of output (default 2)
            state_dim : Dimension of the node states (default 10)
            num_message_passing_rounds : Number of message passing rounds
                (default 3)
        """

    def __init__(self, device, state_dim=32, num_message_passing_rounds=5, eps=1e-10):
        super().__init__()
        self.model_type = 'evidential'
        # Set input dimensions and other hyperparameters
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Define locked parameters
        self.edge_dim = 1
        self.output_dim = 4
        self.num_features = 1
        self.h1_message = 128
        self.h2_message = 256
        self.h3_message = 256
        self.hidden_dim_output = 128

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.num_features, self.h1_message).double(),
            torch.nn.LayerNorm(self.h1_message).double(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.h1_message, self.state_dim).double(),
            torch.nn.LayerNorm(self.state_dim).double(),
            torch.nn.LeakyReLU(),
        )

        # Output net
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim_output).double(),
            torch.nn.Linear(self.hidden_dim_output, self.output_dim).double()
        )

        # Initialize weights
        self.message_net.apply(self.init_weights)
        self.output_net.apply(self.init_weights)

        # Speficy activation functions
        self.softplus = torch.nn.Softplus()

        # Utilize GPU? #TODO: move this out of the class itself and into the run-script
        self.eps = eps
        self.device = device
        self.to(device)

        self.scalar = None

    def init_weights(self,
                     layer):  # Found here: https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        if type(layer) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                          nonlinearity='leaky_relu').double()  # Kaiming for Relu

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
        self.state = torch.zeros([x.num_nodes, self.state_dim]).double().to(self.device)

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            e_len = x.edge_lengths
            e_coulomb = x.edge_coulomb

            # Stacking features
            inp = torch.cat((self.state[x.node_from],
                             e_coulomb,
                             ), 1)
            message = self.message_net(inp)

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).double().to(self.device)
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Get parameters of NIG distribution (4-dimensional output)
        evidential_params_ = self.output_net(self.graph_state)  # (gamma, v, alpha, beta)
        # Apply activations as specified after Equation 10 in the paper
        gamma, v, alpha, beta = torch.tensor_split(evidential_params_, 4, axis=1)

        # if trained on scaled and in eval mode
        if self.scalar is not None and self.training == False:
            # de-scaling prediction
            gamma = torch.from_numpy(self.scalar.inverse_transform(gamma.detach()))

        out = torch.concat(
            [gamma, self.softplus(v) + self.eps, self.softplus(alpha).add(1.0).to(torch.float64) + self.eps,
             self.softplus(beta) + self.eps], axis=1)

        return out



