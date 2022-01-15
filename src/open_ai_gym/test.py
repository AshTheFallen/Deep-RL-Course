import gym

env = gym.make('CartPole-v0')
env.reset()
box = env.observation_space
# print(box)
done = None
counter = 0
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    counter += 1
    print(counter)
