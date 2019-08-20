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
        return samples
        