class Config:
    def __init__(self):
        self.action_size = 1
        self.alpha = 0.001
        self.beta = 0.001
        self.gamma = 0.99
        self.tau = 0.001
        self.actor_h5 = './actor.h5'
        self.critic_h5 = './critic.h5'
        self.state_size = 3
        self.pool_size = 100000
        self.batch_size = 64
        self.num_epochs = 100
        self.update_interval = 20
        self.eval_interval = 20
        self.fc1_dim = 400
        self.fc2_dim = 200
        self.fc1a_dim = 100
        self.dropout = 0.5