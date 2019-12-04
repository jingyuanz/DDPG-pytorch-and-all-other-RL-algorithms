import gym
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import math
from torch import cuda
import torch.nn.functional as F
from config import Config

def init_ff_layer(layer, f1=None):
    weight_size = layer.weight.data.size()[0]
    if not f1:
        f1 = 1 / math.sqrt(weight_size)
    nn.init.uniform_(layer.weight.data, -f1, f1)
    nn.init.uniform_(layer.bias.data, -f1, f1)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.config = Config()
        self.fc1 = nn.Linear(self.config.state_size, self.config.fc1_dim)
        init_ff_layer(self.fc1)
        self.bn1 = nn.LayerNorm(self.config.fc1_dim)
        self.fc2 = nn.Linear(self.config.fc1_dim, self.config.fc2_dim)
        init_ff_layer(self.fc2)
        self.bn2 = nn.LayerNorm(self.config.fc2_dim)
        self.fc3 = nn.Linear(self.config.fc2_dim, self.config.fc3_dim)
        init_ff_layer(self.fc3)
        self.fc_mu = nn.Linear(self.config.fc2_dim, 1)
        self.optimizer = optim.Adam(lr=self.config.alpha, params=self.parameters())
        self.device = self.config.device
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        # x = F.relu(self.fc3(x))
        # fc2_bn = F.dropout(fc2_bn)
        fc_mu = torch.tanh(self.fc_mu(x))
        return self.config.max_action*fc_mu
        
    def save(self):
        torch.save(self.state_dict(), self.config.actor_h5)
    
    def load(self):
        self.load_state_dict(torch.load(self.config.actor_h5))


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.config = Config()
        self.fc1s = nn.Linear(self.config.state_size, self.config.fc1_dim)
        init_ff_layer(self.fc1s)
        self.bn1s = nn.LayerNorm(self.config.fc1_dim)

        self.fc1a = nn.Linear(self.config.action_size, self.config.fc1a_dim)
        init_ff_layer(self.fc1a)
        self.bn1a = nn.LayerNorm(self.config.fc1a_dim)
        
        self.fc2 = nn.Linear(self.config.fc1_dim, self.config.fc2_dim)
        init_ff_layer(self.fc2)
        self.bn2 = nn.LayerNorm(self.config.fc2_dim)
        self.fc3 = nn.Linear(self.config.fc2_dim, self.config.fc3_dim)
        init_ff_layer(self.fc3)
        self.fc_q = nn.Linear(self.config.fc2_dim, 1)
        self.optimizer = optim.Adam(lr=self.config.beta, params=self.parameters())
        self.device = self.config.device
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1s(state))
        # x = self.bn1s(x)
        y = F.relu(self.fc1a(action))
        # y = self.bn1a(y)
        xy = torch.add(x,y)
        z = F.relu(self.fc2(xy))
        # z = F.relu(self.bn2(z))
        # z = F.relu(self.fc3(z))
        # fc2_bn = F.dropout(fc2_bn)
        fc_q = self.fc_q(z)
        return fc_q
    
    def save(self):
        torch.save(self.state_dict(), self.config.critic_h5)
    
    def load(self):
        self.load_state_dict(torch.load(self.config.critic_h5))