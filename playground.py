from agent import *
from config import Config
import gym
import numpy as np

def main():
    config = Config()
    env = gym.make(config.env_name)
    agent = ACAgent()
    scores = []
    prev_best_avg = -100000
    for i in range(config.num_epochs):
        state = env.reset()
        done = False
        epoch_reward = 0.
        while not done:
            action = agent.sample_action(state)
            new_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, new_state, done)
            agent.train()
            state = new_state
            epoch_reward += reward
            env.render()
        scores.append(epoch_reward)
        avg_rewards = np.mean(scores[-config.update_interval:])
        print("epoch : {} avg reward: {}".format(i, avg_rewards))
        if avg_rewards > prev_best_avg:
            agent.save_models()
            print('save models')
            

if __name__ == '__main__':
    main()
        