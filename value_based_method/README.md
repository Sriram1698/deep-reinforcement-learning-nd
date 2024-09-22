# Value-Based Methods

## Description

Traning an agent on [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) environment to navigate and collect bananas in a large, square world.

* The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 

* There are `4` discrete actions that the agent can take on the environment.
    1. `0` - move forward.
    2. `1` - move backward.
    3. `2` - turn left.
    4. `3` - turn right.

* A reward of `+1` is given to the agent if it collects a yellow banana, and a reward of `-1` is provided for collecting a blue banana. The goal of the agent is to collect as many yellow banana as possible while avoiding blue bananas.

* The given task is episodic with fixed time steps of `1000` per episode.

* The environment is solved if the agent is able to get an average score of `+13` over `100` consecutive episodes.

## Dependencies
To setup the environment, follow the following steps.

1. **Setup the python environment**: Follow [the instructions in the GitHub repository](https://github.com/udacity/Value-based-methods#dependencies).

2. **Download the Unity Environment**: Download the Unity environment from one of the links below that matches the operating system:
    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. Place the downloaded file inside `p1_navigation/` folder and unzip the file.

## Training
After the required files are downloaded, open `Navigation.ipynb` file and follow the instructions on each cell and execute it one after the other. 
**`Note`**: Don't forget to replace the file path to the Unity Environment. Suppose if the operating system is Mac OSX, then
```bash
env = UnityEnvironment(file_name="/path/to/Banana.app)
```
