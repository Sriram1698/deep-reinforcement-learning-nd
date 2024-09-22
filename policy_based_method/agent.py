import copy
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.__mu       = mu * np.ones(size)
        self.__theta    = theta
        self.__sigma    = sigma
        self.__seed     = random.seed(seed)
        self.reset()
    
    def reset(self):
        """ 
        Reset the internal state (= noise) to mean (mu).
        """
        self.__state = copy.copy(self.__mu)
    
    def sample(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.__state
        dx = self.__theta * (self.__mu - x) + self.__sigma * np.random.randn(len(x))
        self.__state = x + dx
        return self.__state

class ReplayBuffer:
    """ 
    Fixed-size buffer to store experience tuples.
    """
    def __init__(self, buffer_size, batch_size, device, seed=42):
        """
        Initialize a ReplayBuffer object.
        Parameters
        ----------
            buffer_size (int):  maximum size of buffer
            batch_size (int):   size of each training batch
        """
        self.__device       = device
        self.__memory       = deque(maxlen=buffer_size)
        self.__batch_size   = batch_size
        self.__experience   = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.__seed         = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        e = self.__experience(state, action, reward, next_state, done)
        self.__memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experience from memory.
        """
        experiences = random.sample(self.__memory, k=self.__batch_size)

        states      = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.__device)
        actions     = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.__device)
        rewards     = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.__device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.__device)
        dones       = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.__device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.__memory)
    
class Agent():
    def __init__(self, state_dim, action_dim, gpu=True, seed=42):
        """
        Initialize an Agent object.
        
        Params
        ======
            state_dim (int):    dimension of each state
            action_dim (int):   dimension of each action
            gpu (bool):         true if gpu needs to be utilized
            seed (int):         random seed
        """
        self.__device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if gpu else "cpu"
        self.__state_dim    = state_dim
        self.__action_dim   = action_dim
        self.__seed         = random.seed(seed)

        # Parameters
        self.__lr_actor             = 1e-4                          # learning rate for actor
        self.__lr_critic            = 1e-3                          # learning rate for critic
        self.__gamma                = 0.99                          # discount factor
        self.__tau                  = 1e-3                          # for soft update of target parameters 
        self.__buffer_size          = int(1e5)                      # replay buffer size
        self.__batch_size           = 256                           # minibatch size
        self.__weight_decay         = 0                             # L2 weight decay
        self.__update_every         = 20                            # Update the network every n steps

        # Actor network
        self.__actor_local      = Actor(state_dim, action_dim, seed=seed).to(self.__device)
        self.__actor_target     = Actor(state_dim, action_dim, seed=seed).to(self.__device)
        self.__actor_optimizer  = optim.Adam(self.__actor_local.parameters(), lr=self.__lr_actor)

        # Critic network
        self.__critic_local      = Critic(state_dim, action_dim, seed=seed).to(self.__device)
        self.__critic_target     = Critic(state_dim, action_dim, seed=seed).to(self.__device)
        self.__critic_optimizer  = optim.Adam(self.__critic_local.parameters(), lr=self.__lr_critic, weight_decay=self.__weight_decay)

        # Noise process
        self.__noise    =   OUNoise(action_dim, seed=seed)

        # Replay memory
        self.__memory   = ReplayBuffer(self.__buffer_size, self.__batch_size, device=self.__device, seed=seed)

        # Initialize time step (for updating every "update_freq" steps)
        self.__t_step   = 0

    def step(self, states, actions, rewards, next_states, dones):
        """ Save experience in replay buffer memory, and use random sample from buffer to learn. """
        # Save the experiences
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.__memory.add(state, action, reward, next_state, done)

        # Learn every "update_freq" steps
        self.__t_step = (self.__t_step + 1) % self.__update_every
        
        if (self.__t_step == 0):
            # Learn, if enough samples are available in memory
            if len(self.__memory) > self.__batch_size:
                for _ in range(10):
                    experiences = self.__memory.sample()
                    self.__learn(experiences)
    
    def reset(self):
        self.__noise.reset()
    
    def act(self, states, add_noise=True):
        """ Returns actions for given states as per current policy. """
        states = torch.from_numpy(states).float().to(device=self.__device)

        # Set the actor local network to evaluation mode
        self.__actor_local.eval()
        with torch.no_grad():
            actions = self.__actor_local(states).cpu().data.numpy()

        # Set the network back to training mode
        self.__actor_local.train()

        # Add noise to the output
        if add_noise:
            actions += self.__noise.sample()
        return np.clip(actions, -1, 1)
        
    def __learn(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters
        ----------
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next    = self.__actor_target(next_states)
        Q_targets_next  = self.__critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.__gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.__critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.__critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.__critic_local.parameters(), 1)
        self.__critic_optimizer.step()

        # Update actor
        # Compute actro loss
        actions_pred = self.__actor_local(states)
        actor_loss = -self.__critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.__actor_optimizer.zero_grad()
        actor_loss.backward()
        self.__actor_optimizer.step()

        self.__soft_update(self.__critic_local, self.__critic_target)
        self.__soft_update(self.__actor_local, self.__actor_target)

        self.reset()

    def  __soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters:
        ----------
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.__tau * local_param.data + (1.0 - self.__tau) * target_param.data)

    def save_model(self):
        torch.save(self.__actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(self.__critic_local.state_dict(), 'checkpoint_critic.pth')
    
    def load_model(self, actor_model_file, critic_model_file):
        self.__actor_local.load_state_dict(torch.load(actor_model_file))
        self.__critic_local.load_state_dict(torch.load(critic_model_file))
