# Project 2: Continuous Control 

This repository contains the second project of the [Deep Reinforcement Learning Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), by Udacity.

## Introduction

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

![Training an agent to maintain its position at the target location for as many time steps as possible.](reacher.gif)

### The environment

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). Unity ML-Agents is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

**Note:** The Unity ML-Agent team frequently releases updated versions of their environment. In this repository, the v0.4 interface has been used. The project environment provided by Udacity is similar to, but not identical to the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment on the Unity ML-Agents GitHub page.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

There are two versions of the environment:

* **Version 1: One (1) Agent**  
The task is episodic, and in order to solve the environment, the agent must get an **average score of +30 over 100 consecutive episodes**.

* **Version 2: Twenty (20) Agents**  
The 20 agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). 
At the end of each episode, the rewards that each agent received (without discounting) are added up. 
This generates 20 potentially different scores. Then the average of these 20 scores is taken, which is the average score for each episode (where the average is over all 20 agents).   
Finally the environment is considered solved when the **moving average over 100 episodes** of those average scores **is at least +30**.

In this repository, the second version of the environment has been used with a DDPG algorithm.

## Getting started

### Installation requirements

- Python 3.6 / PyTorch 0.4.0 environment creation: follow the requirements described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
- Clone this repository and have the files accessible in the previously set up Python environment
- For this project, you will not need to install Unity. This is because Udacity has already built the environment for you, and you can download it from one of the links below. You need to only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

- Unzip the environment archive in the project's environment directory and set the path to the UnityEnvironment in the code.

## Instructions

### Training an agent
    
Run the `Continuous_Control.ipynb` notebook and follow the steps in the code.

### Adjusting the Hyperparameters
Here is the list of all the hyperparameters with which you can play and see how the learning change based on them.

**1.** In the **Continuous_Control.ipynb** file  

* n_episodes: Maximum number of training episodes
* maxlen: How many episodes to consider when calculating the moving average
* max_t: Maximum number of time steps per episode
* random_seed: The number used to initialize the pseudorandom number generator

**2.** In the **ddpg_agent.py** file

* BUFFER_SIZE: Replay buffer size
* BATCH_SIZE: Minibatch size
* GAMMA: Discount factor for expected rewards
* TAU: Multiplicative factor for the soft update of target parameters
* LR_ACTOR: Learning rate for the local actor's network
* LR_CRITIC: Learning rate for the local critic's network
* WEIGHT_DECAY: L2 weight decay
* LEARN_EVERY and \LEARN_NUMBER: Update the networks 10 (LEARN_NUMBER) times after every 20 (LEARN_EVERY) timesteps
* EPSILON: Noise factor  
* EPSILON_DECAY: Multiplicative factor for the noise-process rate decay

**3.** In the **model.py** file

* fc1_units and fc2_units: sizes of the actor network's layers
* fc1_units and fc2_units: sizes of the critic network's layers    
