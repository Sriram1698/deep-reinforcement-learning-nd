# Multi-Agent Reinforcement Learning

## Description

Traning two agents (a multi-agent) [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) environment, control rackets to bounce a ball over a net.

* The state space has `24` dimensions corresponding to position and velocity of the ball and racket. Each agent receives its own, local observation.

* Each action is a vector with `2` continuous numbers, corresponding to movement toward (or away from) the net, and jumping. 

* A reward of `+0.1` is provided to an agent, if it hits the ball over the net. A reward of `-0.01` is provided to an agent, if it lets a ball hit the ground or hits the ball out of bounds.

* The given task is episodic with fixed time steps of `2000` per episode.

* The environment is solved if the agent(s) is able to get an average score of `+0.5` and above for over `100` consecutive episodes. 

## Dependencies
To setup the environment, follow the following steps.

1. **Setup the python environment**: Follow [the instructions in the GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).

2. **Download the Unity Environment**: Download the Unity environment from one of the links below that matches the operating system:
    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

3. Place the downloaded file inside `p3_collab-compet/` folder and unzip the file.

## Training
After the required files are downloaded, open `Collab_Compet.ipynb` file and follow the instructions on each cell and execute it one after the other. 
**`Note`**: Don't forget to replace the file path to the Unity Environment. Suppose if the operating system is Mac OSX, then
```bash
env = UnityEnvironment(file_name="/path/to/Tennis.app)
```
