import gym
import numpy as np
from numpy.random import *
import torch
from torch import optim
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
device = torch.device("cuda" if cuda.is_available() else "cpu")
GAMMA = 0.999
LR = 0.001
POOL_SIZE = 10000
EPS_STEP = 500
env = gym.make('CartPole-v0')
START_TRAIN_STEP = 100


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(4, 30)
        self.fc2 = nn.Linear(30, 100)
        self.fc_out = nn.Linear(100, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_out(x))
        return x

qnet = QNet()
qnet2 = QNet()
qnet.double()
qnet2.double()
adam = optim.Adam
def sample_action(s):
    
    if total_step <= EPS_STEP:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            s = torch.tensor([s])
            actions = qnet(s)
            action = actions.argmax(1).item()
    return action


NUM_EPOCH = 1000

memory_pool = []
REPLAY_SIZE = 64

def train_by_replay():
    print("training...")
    samples = choice(range(len(memory_pool)), size=REPLAY_SIZE, replace=False)
    samples = np.asarray(memory_pool)[samples]
    curr_states = [x[0] for x in samples]
    next_states = [x[1] for x in samples]
    curr_states = np.asarray(curr_states)
    next_states = np.asarray(next_states)
    rewards = np.asarray([x[3] for x in samples])
    curr_states = torch.tensor(curr_states)
    next_states = torch.tensor(next_states)
    Q_curr = qnet(curr_states).max(1)
    Q_next = qnet2(next_states).max(1)
    print(Q_next)
    expectedQ = rewards+Q_next*GAMMA
    predQ = Q_curr
    loss = F.smooth_l1_loss(predQ, expectedQ)
    adam.zero_grad()
    loss.backward()
    adam.step()
    

total_step = 0

for epoch in range(NUM_EPOCH):
    print("EPOCH: {}".format(epoch))
    init_state = env.reset()
    curr_state = init_state
    for step in range(200):
        # env.render()
        action = sample_action(curr_state)
        memory = [curr_state]
        state, reward, done, _ = env.step(action) # take a random action
        if done:
            print(1)
            break
        memory += [state, action, reward, done]
        memory = tuple(memory)
        memory_pool.append(memory)
        if len(memory_pool)>POOL_SIZE:
            memory_pool = memory_pool[-POOL_SIZE:]
        if total_step > START_TRAIN_STEP:
            train_by_replay()
        
        curr_state = state
        total_step += 1
    env.close()
    print(step)
    print(total_step)
