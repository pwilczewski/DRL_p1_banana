
# Solution to Project 1: Navigation

### Introduction

This repository contains my solution for the first project in Udacity's Deep Reinforcement Learning Nanodegree. For this project I trained an agent to navigate and collect bananas in a large, square world. In this environment collecting a yellow banana yields a reward of +1 and collecting a blue banana yields a reward of -1. The goal of my agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The simulation contains an agent that navigates a 2-d environment. At each time step, it has four actions at its disposal - move forward, backward, turn left or turn right. These correspond to the action values [0, 1, 2, 3] respectively. The state space has 37 dimensions. It contains data about the agent's velocity and objects in front of the agent. In this environment collecting a yellow banana yields a reward of +1 and collecting a blue banana yields a reward of -1. The agent's objective to maximize the reward by collecting yellow bananas and avoiding build ones. The environment is solved once the agent achieves an average score of +13 over 100 consecutive episodes.

### Getting Started

My solution was coded using Python version 3.6.12, PyTorch version 0.4.0 and OpenAI gym version 0.17.3.

1. The requirements for running my solution are available in the Udacity [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions can be found in `README.md` at the root of the repository.

2. Additionally you will need to download the Banana environment from one of the links below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. After downloading the file you may copy this repository, place the Banana environment file in the project directory and unzip it. 

### Instructions

After completing the initial setup, the Jupyter Notebook `Navigation.ipynb` contains my solution. The notebook references two supporting py files. The `dqn_agent.py` file contains the definition of the agent and the `model.py` file contains the structure of the deep neural network model. Finally the estimated model parameters that solved the environment are located in the `model_weights.pth` file. The `report.md` file contains a descrption of the algorithm used to estimate these parameters.
