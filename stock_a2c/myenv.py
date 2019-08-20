import gym
from gym import spaces
from config import Config
import numpy as np
from numpy.random import randint, rand
from dataloader import DataLoader
class StockEnv(gym.Env):
    def __init__(self, data):
        super(StockEnv,self).__init__()
        self.data = data
        self.config = Config()
        self.len_data = len(data)
        self.reward_range = (0,1)
        self.action_space = spaces.Discrete(self.config.action_size)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)
    
    def reset(self):
        self.start_point = randint(self.config.T, self.len_data-1)
        init_price = self.data[self.start_point]['Closed'].value
        self.fund = randint(self.config.MIN_FUND, self.config.MAX_FUND)
        self.share = randint(self.config.MIN_SHARE, self.config.MAX_SHARE)
        self.net = rand() + 0.5
        self.init_fund = self.net * init_price + self.fund #TODO
        