import gym

env = gym.make('CartPole-v0')
env.reset()

box = env.observation_space

# In [53]: box
# Out[53]: Box(4,)

# In [54]: box.
# box.contains       box.high           box.sample         box.to_jsonable
# box.from_jsonable  box.low            box.shape

env.action_space

# In [71]: env.action_space
# Out[71]: Discrete(2)

# In [72]: env.action_space.
# env.action_space.contains       env.action_space.n              env.action_space.to_jsonable
# env.action_space.from_jsonable  env.action_space.sample

done = False
step_count = 0
while not done:
    observation, reward, done, _ = env.step(env.action_space.sample())
    step_count += 1

print(step_count)
