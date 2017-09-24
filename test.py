import gym
env = gym.make('Tennis-ram-v0')
env.reset()

print('Actions space: {}'.format(env.action_space))

for i_episode in range(1):
    observation = env.reset()
    print('Episode: {}'.format(i_episode))
    done_n = False
    t = 0
    while done_n == False:
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward_n, done_n, info_n = env.step(action)
        t = t + 1
        if reward_n != 0.0:
            print('Reward: {}, done: {}, info: {}'.format(reward_n, done_n, info_n))
        if done_n:
            print("Episode finished after {} timesteps".format(t + 1))
            break