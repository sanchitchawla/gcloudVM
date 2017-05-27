import gym 
import numpy as np 

env = gym.make('FrozenLake-v0')

# initialize table with zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Learning parameters

lr = .8
y = .95
num_episodes = 10000

# create list to contain total rewards and steps per episode
reward_lis = []

for i in range(num_episodes):
	# reset env to get new observation
	s = env.reset()
	rAll = 0
	d = False
	j = 0

	# Q learning 
	while j < 99:
		j+=1

		# choose an action with high greed

		action = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))

		# get the new state and reward from env
		#print action
		s1,reward,done, _ = env.step(action)

		# Update q table
		Q[s,action] = Q[s,action] + lr*(reward + y*np.max(Q[s1,:]) - Q[s,action])
		rAll += reward
		s = s1
		if done:
			break
	reward_lis.append(rAll)

print "Score over time: " + str(sum(reward_lis) / num_episodes)
#print "Q table", Q