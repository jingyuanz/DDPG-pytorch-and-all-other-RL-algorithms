class Config:
    def __init__(self):
        self.action_size = 1
        self.alpha = 0.0001
        self.beta = 0.05
        self.gamma = 0.999
        self.tau = 0.1
        self.actor_h5 = './model/actor.h5'
        self.critic_h5 = './model/critic.h5'
        self.target_actor_h5 = './model/target_actor.h5'
        self.target_critic_h5 = './model/target_critic.h5'
        self.state_size = 2
        self.pool_size = 100000
        self.batch_size = 128
        self.num_epochs = 1000
        self.update_interval = 20
        self.eval_interval = 1
        self.fc1_dim = 100
        self.fc2_dim = 50
        self.fc1a_dim = 100
        self.fc3_dim = 20
        self.dropout = 0.5
        self.env_name = 'MountainCarContinuous-v0'
        self.max_action = 1