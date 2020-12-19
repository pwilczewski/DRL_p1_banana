import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Define layers
        self.fc1 = nn.Linear(state_size,8)
        self.fc2 = nn.Linear(8,16)
        self.fc3 = nn.Linear(16,action_size)

    def forward(self, state):
        
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.fc3(state)
        state = F.relu(state)
        
        return state
