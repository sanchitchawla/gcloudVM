import gym
from gym import wrappers
import numpy as np
import mdptoolbox, mdptoolbox.example

bins = [[-0.77038577, -0.01766937],
[-0.78736876, -0.01698298],
[-0.80357355, -0.0162048 ],
[-0.84076519, -0.01143178]]

num_states = (len(bins[0])+1)**len(bins)
P,_ = mdptoolbox.example.rand(num_states, 3)
R = np.ones((num_states,))
reward_sum = []

def normalize(m):
    for j in range(m.shape[0]):
        for i in range(m.shape[1]):
            m[j,i,:] = m[j,i,:] / m[j,i,:].sum()
    return m

def obs2state(obs):
    ans = 0;
    for i in range(len(obs)):
        k = np.digitize(obs[i],bins[i])
        ans = ans*(len(bins[0])+1) + k
    return ans

all_obs = []
perf = []

name = 'MountainCar-v0'
#name = 'CartPole-v0'
env = gym.make(name)
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-3')

for i_episode in range(1000):
    observation = env.reset()
    action = env.action_space.sample()
    r = 0
    for t in range(200):

        env.render()

        from_state = obs2state(observation)
        #print observation

        observation, reward, done, info = env.step(action)
        #print action
        all_obs.append(observation)

        to_state = obs2state(observation)
        #print from_state, to_state, "STATES"
        r += reward
        if done:
            #print R
            if t == 199:
                R[to_state] = -1000.0
            else:
                R[to_state] = 1000
            print("{}:Episode finished after {} timesteps".format(i_episode, t+1))
            perf.append(t+1)
            break


        if t < 170:
            # only retrain if below target
            P[action,from_state,to_state] += 0.50
            #print P, "P"
            P = normalize(P)
            H = min(200-t,200)
            mdp = mdptoolbox.mdp.FiniteHorizon(P,R,0.9,H)
            #mdp = mdptoolbox.mdp.QLearning(P,R,0.9)
            #mdp = mdptoolbox.mdp.PolicyIterationModified(P, R, 0.99, epsilon=0.001, max_iter=300)
            mdp.run()
        if t == 198 and not done:
            # only retrain if below target
            P[action,from_state,to_state] *= 0.000001
            P = normalize(P)
            H = min(200-t,200)
            mdp = mdptoolbox.mdp.FiniteHorizon(P,R,0.9,H)
            #mdp = mdptoolbox.mdp.QLearning(P,R,0.9)
            #mdp = mdptoolbox.mdp.PolicyIterationModified(P, R, 0.99, epsilon=0.001, max_iter=300)
            mdp.run()

        if t < 199 and done:
            "YES IT SOLVED ONCE"
            P[action,from_state,to_state] += 100
            print P, "P"
            P = normalize(P)
            H = min(200-t,200)
            mdp = mdptoolbox.mdp.FiniteHorizon(P,R,0.9,H)
            #mdp = mdptoolbox.mdp.QLearning(P,R,0.9)
            #mdp = mdptoolbox.mdp.PolicyIterationModified(P, R, 0.99, epsilon=0.001, max_iter=300)
            mdp.run()
            break
        #action = int(mdp.policy[to_state])
        #action = env.action_space.sample()
        action = mdp.policy[to_state,0]
    reward_sum.append(r)
    #print "Reward for episode {} is {}".format(i_episode, reward_sum[-1])
