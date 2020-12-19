import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        # Calls functions / attributes of whatever the parent class is
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        
        # Define layers
        self.fc1 = nn.Linear(state_size,8)
        self.fc2 = nn.Linear(8,16)
        
        # Four output actions
        self.fc4 = nn.Linear(16,action_size)
        
        # using Sequential instead?
        # self.model = nn.Sequential(nn.Linear(state_size,32),nn.ReLU() ... )

    # state is a tensor, pass it through each of operations
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # Apply operations to state - it's all ReLU to me!
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.fc4(state)
        # Generate output layer 
        state = F.relu(state)
        
        return state
