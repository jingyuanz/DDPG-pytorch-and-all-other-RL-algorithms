from config import Config
from model import *
from replay_buffer import ReplayBuffer

class OUActionNoise:
    def __init__(self, mu, sigma=0.25, theta=.1, dt=1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)



class ACAgent:
    def __init__(self):
        self.actor = Actor()
        self.target_actor = Actor()
        self.critic = Critic()
        self.target_critic = Critic()
        self.replay_buffer = ReplayBuffer()
        self.config = Config()
        self.soft_update(self.target_actor, self.actor, True)
        self.soft_update(self.target_critic, self.critic, True)
        self.ou = OUActionNoise(mu=0.)
        self.device = self.actor.device

    def to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def soft_update(self, target, local, hard=False):
        if hard:
            tau = 1
        else:
            tau = self.config.tau
        for ta, lo in zip(target.parameters(), local.parameters()):
            ta.data.copy_(ta.data*(1-tau)+lo.data*tau)
    
    def sample_action(self, state):
        state = self.to_tensor(state)
        noise = self.ou()
        with torch.no_grad():
            self.actor.eval()
            mu = self.actor(state) + noise
            mu_cpu = mu.cpu()
        return mu_cpu
    
    def train_critic(self, loss):
        self.critic.optimizer.zero_grad()
        loss.backward()
        self.critic.optimizer.step()

    def train_actor(self, loss):
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()

    def update_all_targets(self):
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def save_models(self):
        self.target_critic.save()
        self.target_actor.save()
        self.critic.save()
        self.actor.save()
    
    def load_models(self):
        self.target_critic.load()
        self.target_actor.load()
        self.critic.load()
        self.actor.load()

    def train(self):
        if len(self.replay_buffer.pool) < self.config.batch_size*5:
            return
        statesT, actions, rewards, statesTp1, done_flags = self.replay_buffer.sample_batch()
        statesT = self.to_tensor(statesT)
        statesTp1 = self.to_tensor(statesTp1)
        rewards = self.to_tensor(rewards)
        rewards = rewards.view(self.config.batch_size, 1)

        actions = self.to_tensor(actions)
        actions = actions.view(self.config.batch_size, 1)

        done_flags = self.to_tensor(done_flags)
        done_flags = done_flags.view(self.config.batch_size, 1)
        self.target_critic.eval()
        self.target_actor.eval()
        with torch.no_grad():
            actions_tp1 = self.target_actor(statesTp1)
            Q_tp1 = self.target_critic(statesTp1, actions_tp1)
            
        y = rewards + self.config.gamma * done_flags * Q_tp1
        # train critic
        self.critic.train()
        Q = self.critic(statesT, actions)
        critic_loss = F.mse_loss(y, Q)
        self.train_critic(critic_loss)
        
        # train actor
        self.critic.eval()
        self.actor.train()
        actions_pred = self.actor(statesT)
        Q_actor = self.critic(statesT, actions_pred)
        actor_loss = -torch.mean(Q_actor)
        self.train_actor(actor_loss)
        
        self.update_all_targets()
        
        
    
            
if __name__ == '__main__':
    agent = ACAgent()
    agent.soft_update(agent.target_actor, agent.actor, True)