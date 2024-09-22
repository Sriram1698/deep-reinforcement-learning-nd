# Policy-Based Methods

## Description

Traning a double-jointed arm agent on [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) environment to move to different target locations.

* The state space has `33` dimensions corresponding to position, rotation, velocity, and angular velocities of the arm.. 

* Each action is a vector with `4` numbers, corresponding to torque applicable to two joints. Every entry in the action vector is between `-1` and `1`.

* A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps are possible.

* The given task is episodic with fixed time steps of `1000` per episode.

* The environment is solved if the agent(s) is able to get an average score of `+30` and above for over `100` consecutive episodes. 

* In this problem, we took the second version, which is `20 agents`, each with its own copy of the environment.

## Dependencies
To setup the environment, follow the following steps.

1. **Setup the python environment**: Follow [the instructions in the GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).

2. **Download the Unity Environment**: Download the Unity environment from one of the links below that matches the operating system:
    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    **`Note`**: The above links are only for the second (`20 agents`) version.

3. Place the downloaded file inside `p2_continuous-control/` folder and unzip the file.

## Training
After the required files are downloaded, open `Continuous_Control.ipynb` file and follow the instructions on each cell and execute it one after the other. 
**`Note`**: Don't forget to replace the file path to the Unity Environment. Suppose if the operating system is Mac OSX, then
```bash
env = UnityEnvironment(file_name="/path/to/Reacher.app)
```
