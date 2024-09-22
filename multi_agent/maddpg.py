import torch
import numpy as np
from agent import Agent
from collections import deque
from unityagents import UnityEnvironment

class MADDPG():
    def __init__(self, state_dim, action_dim, brain_name, num_agents, gpu=True, seed=42):
        self.__state_dim    = state_dim
        self.__action_dim   = action_dim
        self.__num_agents   = num_agents
        self.__seed         = seed
        self.__brain_name   = brain_name
        self.__device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if gpu else "cpu"
        self.__n_agents     = [Agent(id, state_dim, action_dim, num_agents, gpu, seed) for id in range(0, num_agents)]
    
    def reset(self):
        for agent in self.__n_agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones):
        for reward, done, agent in zip(rewards, dones, self.__n_agents):
            agent.step(np.expand_dims(states, axis=0), np.expand_dims(actions, axis=0), reward, np.expand_dims(next_states, axis=0), done)

    def act(self, states):
        actions = []
        for agent in self.__n_agents:
            actions.append(agent.act(states))
        
        # Return joint actions (collective actions of all the agents)
        actions = np.vstack(actions)
        return actions

    def save_model(self):
        for agent in self.__n_agents:
            agent.save_model()

    def load_model(self):
        for agent in self.__n_agents:
            agent.load_model()

    def train(self, env: UnityEnvironment, num_episodes = 5000, max_t = 2000, print_every=100):
        scores = []
        scores_window= deque(maxlen = print_every)

        for i_episode  in range(1, num_episodes + 1):
            env_info = env.reset(train_mode=True)[self.__brain_name]
            states = env_info.vector_observations
            score = np.zeros(self.__num_agents)
            self.reset()

            for _ in range(max_t):
                actions     = self.act(states)
                env_info    = env.step(actions)[self.__brain_name]
                next_states = env_info.vector_observations
                rewards     = env_info.rewards
                dones       = env_info.local_done
                self.step(states, actions, rewards, next_states, dones)
                score       += rewards
                states      = next_states 
                if any(dones):
                    break

            scores.append(np.max(score))
            scores_window.append(np.max(score))
            print('\rEpisode {}\tScore: {:.4f}\tAverage Score: {:.4f}'.format(i_episode, np.max(score), np.mean(scores_window)), end='')
            if i_episode % print_every == 0:
                self.save_model()
                print('\rEpisode {}\tScore: {:.4f}\tAverage Score: {:.4f}'.format(i_episode, np.max(score), np.mean(scores_window)))
                    
        return scores