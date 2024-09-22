import numpy as np

def hidden_layer_param_initializer(layer):
    """ 
    Reset the layer's parameters with the input dimension of the layer
    Parameters
    ----------
        layer: The layer for which the weights needs reset
    """
    in_units = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(in_units)
    return (-lim, lim)