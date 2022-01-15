import gym
import pyglet
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange


def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, params):
    env.render()
    observation = env.reset()
    done = False
    t = 0
    # will play 200 steps at max
    while not done:
        t += 1
        action = get_action(observation, params)
        observation, reward, done, _ = env.step(action)
        if done:
            break
    return t


def play_multiple_episodes(env, params, episodes):
    episodes_steps = np.zeros(episodes)
    for i in range(episodes):
        episodes_steps[i] = play_one_episode(env, params)
    return episodes_steps.mean()


def random_search(env):
    episode_length = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4) * 2 - 1
        episode_length.append(play_multiple_episodes(env, new_params, 100))
        if episode_length[t] > best:
            best = episode_length[t]
            params = new_params
    return episode_length, params


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.reset()
    episode_length, params = random_search(env)
    plt.plot(episode_length)
    plt.show()
