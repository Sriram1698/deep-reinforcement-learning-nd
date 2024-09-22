import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class Actor(nn.Module):
    """ Actor model """
    def __init__(self, state_size, action_size, seed=42):
        """
        Initialize the Actor architecture parameters
        Parameters
        ----------
            layers (list):  A list contains the size of all the layers (including input and output layers) 
            seed (int):     Random seed
        """
        super(Actor, self).__init__()
        self.__seed         = torch.manual_seed(seed)

        # Initialize the model 
        self.__fc1 = nn.Linear(state_size, 512)
        self.__fc2 = nn.Linear(512, 256)
        self.__out = nn.Linear(256, action_size)

        # Reset all the parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """ 
        Reset all the learnable parameters. 
        """
        self.__fc1.weight.data.uniform_(*utils.hidden_layer_param_initializer(self.__fc1))
        self.__fc2.weight.data.uniform_(*utils.hidden_layer_param_initializer(self.__fc2))
        self.__out.weight.data.uniform_(-3e-3, 3e-3)
            
    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        """
        x = F.relu(self.__fc1(state))
        x = F.relu(self.__fc2(x))
        return torch.tanh(self.__out(x))
        
    
class Critic(nn.Module):
    """ Critic model """

    def __init__(self, state_size, action_size, seed=42, fcs1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int):   Dimension of each state
            action_size (int):  Dimension of each action
            seed (int): Random  seed
            fcs1_units (int):   Number of nodes in the first hidden layer
            fc2_units (int):    Number of nodes in the second hidden layer
            fc3_units (int):    Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.__seed = torch.manual_seed(seed)
        self.__fcs1 = nn.Linear(state_size + action_size, fcs1_units)
        self.__fc2  = nn.Linear(fcs1_units, fc2_units)
        self.__out  = nn.Linear(fc2_units, 1)
        # Reset the weight parameters
        self.reset_parameters()

    def reset_parameters(self):
        """ 
        Reset all the learnable parameters. 
        """
        self.__fcs1.weight.data.uniform_(*utils.hidden_layer_param_initializer(self.__fcs1))
        self.__fc2.weight.data.uniform_(*utils.hidden_layer_param_initializer(self.__fc2))
        self.__out.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        """ 
        Build a critic (value) network that maps (state, action) pairs -> Q-values.
        """
        xs  = torch.cat((state, action), dim=1)
        x   = F.relu(self.__fcs1(xs))
        x   = F.relu(self.__fc2(x))
        return self.__out(x)
        
