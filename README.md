[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Solution to Project 1: Navigation

### Introduction

This repository contains my solution for the first product in Udacity's Deep Reinforcement Learning Nanodegree. For this project I trained an agent to navigate and collect bananas in a large, square world.

![Trained Agent][image1]

In this environment collecting a yellow banana yields a reward of +1 and collecting a blue banana yields a reward of -1. The goal of my agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic and the environment is considered solve once the agent achieves an average score of +13 over 100 consecutive episodes.

### Getting Started

My solution was coded using Python version 3.6.12, PyTorch version 0.4.0 and OpenAI gym version 0.17.3.

1. The requirements for running my solution are available in the Udacity [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

2. Additionally you will need to download the Banana environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. After downloading the file you may check out this repository, place the Banana environment file in the folder and unzip it. 

### Instructions

After completing the initial setup, the Jupyter Notebook `Navigation.ipynb` contains my solution. The notebook references two supporting py files. The `dqn_agent.py` file contains the definition of the agent and the `model.py` file contains the structure of the deep neural network model. 
