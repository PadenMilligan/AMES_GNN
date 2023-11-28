
import torch
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, Sequential

# from device import device

def graph_potential(
    n_gc_layers: int = 1,
    n_fc_layers: int = 1,
    activation: str = "Tanh",
    pooling: str = "add",
    n_node_features: int = 55,
    n_edge_features: int = 1,
    n_neurons: int = None,
) -> torch.nn.Module:

    """
    A function that returns a model with the desired number of
    Graph-Convolutional (n_gc_layers) layers and Fully-Connected
    (n_fc_layers) linear layers, using the specified non-linear
    activation layers interspaced between them.

    Args:

    :param int n_gc_layers: number of Graph-Convolution layers (CGConv); default: 1
    :param int n_fc_layers: number of densely-connected layers (n_fc_layers); default: 1
    :param str activation: nonlinear activation layer; can be any activation
          available in torch; default: Tanh
    :param str pooling: the type of pooling of node results; can be 'add',
          i.e. summing over all nodes (e.g. to return total energy; default)
          or 'mean', to return energy per atom.
    :param int n_node_features: length of node feature vectors; default: 55
    :param int n_edge_features: length of edge feature vectors; currently 1,
          the distance.
    :param int n_neurons: number of neurons in deep layers (if they exist)
          in the densely-connected network if set to None, it is internally
          set to n_node_features

    """

    activation_layer = eval("torch.nn." + activation + "()")

    if pooling == "add":
        pooling_layer = (global_add_pool, " y, batch -> energy ")
    else:
        pooling_layer = (global_mean_pool, " y, batch -> energy ")

    if n_neurons is None:
        n_neurons = n_node_features

    layers = []

    # first the Graph-convolutional layers:

    for n_layer in range(0, n_gc_layers):

        if n_layer == 0:

            vals = " x, edge_index, edge_attr -> x0 "

        else:

            vals = " x" + repr(n_layer - 1) + \
              " , edge_index, edge_attr -> x" + repr(n_layer)

        new_layer = (CGConv(n_node_features, dim=n_edge_features), vals)
        layers.append(new_layer)

        vals = " x" + repr(n_layer) + " -> x" + repr(n_layer)
        new_activation = (activation_layer, vals)
        layers.append(new_activation)

    # then the fully-connected layers:

    if n_fc_layers == 1:  # if only one fully connected layer

        if n_gc_layers > 0:

            vals = " x" + repr(n_gc_layers - 1) + " -> y "

        else:

            vals = " x -> y "

        new_layer = (torch.nn.Linear(n_node_features, 1), vals)
        layers.append(new_layer)

    else:

        if n_gc_layers > 0:

            vals = " x" + repr(n_gc_layers - 1) + " -> y0 "

        else:

            vals = " x -> y0 "

        new_layer = (torch.nn.Linear(n_node_features, n_neurons), vals)
        layers.append(new_layer)

        vals = " y0 -> y0 "
        new_activation = (activation_layer, vals)

        layers.append(new_activation)

        for n_layer in range(1, n_fc_layers - 1):

            vals = " y" + repr(n_layer - 1) + " -> y" + repr(n_layer)
            new_layer = (torch.nn.Linear(n_neurons, n_neurons), vals)
            layers.append(new_layer)

            vals = "y" + repr(n_layer) + " -> y" + repr(n_layer)
            new_activation = (activation_layer, vals)
            layers.append(new_activation)

        # and finally the exit layer

        vals = " y" + repr(n_fc_layers - 2) + " -> y "
        new_layer = (torch.nn.Linear(n_neurons, 1), vals)
        layers.append(new_layer)

    # last but not least, the pooling layer

    layers.append(pooling_layer)

    # create and return the model

    model = Sequential(" x, edge_index, edge_attr, batch ", layers)

    return model
