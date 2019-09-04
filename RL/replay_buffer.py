from config import Config
import numpy as np
from random import sample

class ReplayBuffer:
    def __init__(self):
        self.config = Config()
        self.pool = []
        
    def clear_pool(self):
        self.pool = []
    
    def push(self, st, at, rt, stp1, done):
        self.pool.append((st, at, rt, stp1, done))
        if len(self.pool) > self.config.pool_size:
            self.pool = self.pool[-self.config.pool_size:]
    
    def sample_batch(self):
        batch_size = self.config.batch_size
        samples = sample(self.pool, batch_size)
        statesT = []
        statesTp1 = []
        actions = []
        rewards = []
        done_flags = []
        for st, at, rt, stp1, done in samples:
            statesT.append(st)
            actions.append(at)
            rewards.append(rt)
            statesTp1.append(stp1)
            if done:
                done_flags.append(0)
            else:
                done_flags.append(1)
        return statesT, actions, rewards, statesTp1, done_flags
        