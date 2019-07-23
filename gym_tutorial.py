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
TARGET_UPDATE = 10
MIN_EPS = 0.05
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc_out = nn.Linear(30, 2)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(100)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.bn2(x)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = F.softmax(self.fc_out(x))
        return x

qnet = QNet()
qnet2 = QNet()
qnet.double()
qnet2.double()
qnet2.load_state_dict(qnet.state_dict())
qnet2.eval()
adam = optim.RMSprop(qnet.parameters(), lr=LR, weight_decay=1e-4)
eps = 1.
def sample_action(s):
    if total_step <= EPS_STEP:
        action = env.action_space.sample()
    else:
        eps = 1. - 0.05*(total_step//150)
        if total_step % 1000 == 0:
            print(eps)
        if np.random.rand()<eps:
            return env.action_space.sample()
        qnet.eval()
        with torch.no_grad():
            s = torch.tensor([s])
            actions = qnet(s)
            action = actions.max(1)[1].item()
    
    return action


NUM_EPOCH = 1000

memory_pool = []
REPLAY_SIZE = 64

def train_by_replay():
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
    q_next = pred_next.max(1)[0]
    
    # print(action_next)
    # print(q_next)
    
    # print(done.size(), pred_next.size())
    Q_next = (done.double() * q_next)
    # print(pred_next)
    # action_next = pred_next[1]
    # print(action_curr)
    # print(action_next)
    expectedQ = rewards.double()+Q_next*GAMMA
    # loss = F.smooth_l1_loss(predQ, expectedQ)
    loss = (expectedQ - Q_curr).pow(2).mul(0.5).mean()
    adam.zero_grad()
    loss.backward()
    adam.step()
    

total_step = 0

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
        action = sample_action(curr_state)
        memory = [curr_state]
        state, reward, done, _ = env.step(action) # take a random action
        memory += [state, [action], reward, done]
        memory = tuple(memory)
        memory_pool.append(memory)
        if len(memory_pool)>POOL_SIZE:
            memory_pool = memory_pool[-POOL_SIZE:]
        if total_step > START_TRAIN_STEP:
            train_by_replay()
        if done:
            print(step)
            break
        curr_state = state
        total_step += 1
    if epoch % TARGET_UPDATE == 0:
        print('update model')
        qnet2.load_state_dict(qnet.state_dict())
        torch.save(qnet.state_dict(), './model.h5')
    env.close()
