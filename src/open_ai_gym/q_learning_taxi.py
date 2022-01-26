import numpy as np
import gym
import random
import time

env = gym.make("Taxi-v3")
env.render()
state_space = env.observation_space.n
action_space = env.action_space.n
Q = np.zeros((state_space, action_space))

total_episodes = 25000  # Total number of training episodes
total_test_episodes = 100  # Total number of test episodes
max_steps = 200  # Max steps per episode

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.001  # Minimum exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration prob


def epsilon_greedy(Q, state):
    if random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    else:
        return env.action_space.sample()


# Q-learning algorithm
for e in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * e)
    for step in range(max_steps):
        action = epsilon_greedy(Q, state)
        next_state, reward, done, info = env.step(action)
        Q[state][action] = Q[state][action] + learning_rate * (reward + gamma *
                                                               np.max(Q[next_state]) - Q[state][action])
        if done:
            break
        state = next_state

print('training is done')

rewards = []

frames = []
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)
    for step in range(max_steps):
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Q[state][:])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            # print ("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))