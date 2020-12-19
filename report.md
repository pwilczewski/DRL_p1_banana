# Project 1: Report

### Implementation

Description of implementation.

### Learning Algorithm

Description of learning algorithm, chosen hyperparameters and architecture of neural network. 

The hyperparameters for the algorithm are:

BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LR = 0.001
UPDATE_EVERY = 4
EPS = 1.0
EPS_DECAY = 0.99
EPS_MIN = 0.01

Agent takes epsilon-greedy actions.
Uses mean squared error as the loss function.
Adam as the optimizer.

The neural network takes a state of size 37 as input. The first layer contains 8 nodes with ReLU activation and the second layer contains 16 nodes with ReLU activation. The neural network outputs four action values using ReLU activation. In total this architecture contains XXXX parameters.

### Plot of Rewards

After 555 episodes, the successful agent was able to achieve an average score of +13 over its last 100 episodes.

![Score history](score_history.png)

### Ideas for Future Work

Feature engineering.
Alternative neural network architectures.
