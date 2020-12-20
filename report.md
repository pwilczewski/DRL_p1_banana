# Project 1: Report

### Implementation

The solution is implemented as a Deep-Q Network with experience replay. The `Agent` class defines an agent that can uses data about the environment's state to choose an action and then learn from its experience. The `QNetwork` class defines the deep neural network that informs the agent's choices. The `ReplayBuffer` class stores past experiences that the agent samples from to update the model parameters. The `play_dqn()` function trains the agent and outputs scores for each episode.

### Learning Algorithm

The purpose of the learning algorithm is to parameterize a deep neural network that approximates the action-value (Q-value) function for the agent in the Banana environment. After initialization the agent interacts with the environment over a series of episodes. In each episode the agent considers the state of the environment, takes actions and receives rewards. Using this data over many episodes the agent updates its estimate of the action-value for each possible action given data about the state of the environment. 

The agent interacts with the environment by taking epsilon-greedy actions. Initially it acts randomly with `epsilon = 1` but this value decays geometrically at a rate of 1% per episode until reaching a floor of `min(epsilon, 0.01)`. Each action updates the state of the environment and returns some reward. Then each experience observed by the agent is stored in the replay buffer. Every 4 steps through the environment the agent samples 64 experiences from the replay buffer and updates the parameters of the deep neural network.

To update the network parameters the agent uses stochastic gradient descent with the Adam optimizer and a learning rate equal to `0.001`. To calculate the loss it applies a modified version of the Bellman equation. The objective is to minimize the mean squared error between the action values predicted by the network and an estimate of the true action values. The true action values are estimated as the current reward plus the value of future expected rewards, discounted by `0.99`. To avoid getting stuck in a local minimium, the agent stores two copies of the deep neural network. The first `qnetwork_local` is used to predict the action values for the current state. The second `qnetwork_target` is used to predict the action values of the next state. After calculating the gradient of the error the optimizer takes a step in the direction of the gradient and updates the parameters of the `qnetwork_local` network. Finally the agent updates the parameters of the `qnetwork_target` network as a weighted-average of the two networks with `tau = 0.001`.

The deep neural network takes a state of size 37 as input. The first layer contains 8 nodes with ReLU activation and the second layer contains 16 nodes with ReLU activation. The network outputs four action values using ReLU activation. In total this architecture contains 516 parameters.

### Plot of Rewards

After 555 episodes, the successful agent was able to achieve an average score of +13 over its last 100 episodes.

![Scores](score_history.png)

### Ideas for Future Work

One idea for future work is to test alternative network architecures. Additional nodes or additional layers could improve the agent's performance by utilizing more data from the environment. Another idea for future work is to test feature design. For example applying a heuristic like 'stay away from corners and walls' may help the agent learn to collect bananas faster.
