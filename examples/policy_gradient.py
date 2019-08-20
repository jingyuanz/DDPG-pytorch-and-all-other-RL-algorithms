import gym
import numpy as np
from numpy.random import *
import torch
from torch import optim
import torch.nn as nn
import math
from torch import cuda
import torch.nn.functional as F

device = torch.device("cuda" if cuda.is_available() else "cpu")



class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.fc2 = nn.Linear(300, 32)
        self.fc3 = nn.Linear(32, 4)
        self.fc_out = nn.Linear(4, 2)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(30)
    
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        x = self.fc_out(x)
        return x


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.fc2 = nn.Linear(300, 32)
        self.fc3 = nn.Linear(32, 4)
        self.fc_out = nn.Linear(4, 2)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(30)
    
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        x = self.fc_out(x)
        return x
    

def soft_update(local_model, target_model):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


class PolicyGradientAgent:
    def __init__(self):
        self.params = {
            "GAMMA": 0.9999,
            'LR' : 0.001,
            'POOL_SIZE': 100000,
            'EPS_STEP': 1000,
            'START_TRAIN_STEP': 500,
            'TARGET_UPDATE':10,
            'MIN_EPS': 0.01,
            'epsilon_decay': 0.99999,
            'TAU': 8e-2
        }
        self.env = gym.make('CartPole - v0')
        self.actor_net =

qnet = QNet()
qnet2 = QNet()
qnet.double()
qnet2.double()
qnet2.load_state_dict(qnet.state_dict())
qnet2.eval()
adam = optim.Adam(qnet.parameters(), lr=LR)


def sample_action(epoch, eps, s):
    # if epoch > 50:
    #     eps = 0.01
    if total_step <= EPS_STEP:
        action = env.action_space.sample()
    else:
        if total_step % 1000 == 0:
            print(eps)
        if np.random.rand() < eps:
            return env.action_space.sample()
        # qnet.eval()
        with torch.no_grad():
            s = torch.tensor([s])
            actions = qnet2(s)
            action = actions.max(1)[1].item()
    
    return action


NUM_EPOCH = 1000

memory_pool = []
REPLAY_SIZE = 64


def train_by_replay(eps, doubleQ=False):
    # qnet.train()
    samples = choice(range(len(memory_pool)), size=REPLAY_SIZE, replace=False)
    samples = np.asarray(memory_pool)[samples]
    curr_states = [x[0] for x in samples]
    next_states = [x[1] for x in samples]
    done = [x[-1] for x in samples]
    done = torch.tensor([0 if x else 1 for x in done])
    executed_actions = torch.tensor([x[2] for x in samples])
    curr_states = np.asarray(curr_states)
    next_states = np.asarray(next_states)
    rewards = torch.tensor([x[3] for x in samples])
    curr_states = torch.tensor(curr_states)
    next_states = torch.tensor(next_states)
    pred = qnet(curr_states)
    Q_curr = pred.gather(1, executed_actions)
    
    # Q_curr = pred_curr[0]
    # action_curr = pred_curr[1]
    pred_next = qnet2(next_states).detach()
    action_next = pred_next.max(1)[1]
    # print(action_next)
    if doubleQ:
        q_next_pred_action = qnet(next_states).max(1)[1]
        q_next = torch.index_select(pred_next, 1, q_next_pred_action)
    
    else:
        q_next = pred_next.max(1)[0]
    
    # print(action_next)
    # print(q_next)
    
    # print(done.size(), pred_next.size())
    Q_next = (done.double() * q_next)
    # print(pred_next)
    # action_next = pred_next[1]
    # print(action_curr)
    # print(action_next)
    expectedQ = rewards.double() + Q_next * GAMMA
    # loss = F.smooth_l1_loss(Q_curr, expectedQ)
    loss = (expectedQ - Q_curr).pow(2).mul(0.5).mean()
    loss.clamp(-1, 1)
    adam.zero_grad()
    loss.backward()
    adam.step()
    # print(loss)
    if eps > MIN_EPS:
        eps *= epsilon_decay
    return eps


total_step = 0
eps = 1.

for epoch in range(NUM_EPOCH):
    print("EPOCH: {}".format(epoch))
    init_state = env.reset()
    curr_state = init_state
    render = False
    if epoch % 100 == 0:
        render = True
    for step in range(200):
        if render:
            env.render()
        action = sample_action(epoch, eps, curr_state)
        memory = [curr_state]
        state, reward, done, _ = env.step(action)  # take a random action
        memory += [state, [action], reward, done]
        memory = tuple(memory)
        memory_pool.append(memory)
        if len(memory_pool) > POOL_SIZE:
            memory_pool = memory_pool[-POOL_SIZE:]
        total_step += 1
        if done:
            print(step)
            break
        curr_state = state
        if total_step > START_TRAIN_STEP:
            eps = train_by_replay(eps, doubleQ=False)
    if epoch % TARGET_UPDATE == 0:
        print('update model')
        # soft_update(qnet, qnet2)
        qnet2.load_state_dict(qnet.state_dict())
        # torch.save(qnet.state_dict(), './model.h5')
    # env.close()
