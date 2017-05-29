import gym
import numpy as np 
import random
import tensorflow as tf 
import matplotlib.pyplot as plt 

# Hyper parameters
learning_rate = 0.1
y = .99
e = 0.1
num_episodes = 2000

# Lists to store total rewards and steps per episode
jList = []
reward_lis = [] 

# Get the env
env = gym.make('FrozenLake-v0')

# Q network

tf.reset_default_graph()

# feed forward one layer network
inputs1 = tf.placeholder(shape= [1,16], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Loss function sum of squared error
nextQ = tf.placeholder(shape=[1,4], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate)
updateModel = trainer.minimize(loss)

# Training the network

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):

        # Reset env and get observation
        s = env.reset()
        rAll = 0
        j = 0

        # Q network

        while j < 99:
            j+=1

            # Choose an action greedily with e probability
            action, allQ = sess.run([predict, Qout], feed_dict = {inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                action[0] = env.action_space.sample()

            # Get state and reward
            s1, reward, done, _ = env.step(action[0])

            # Obtain the Q values by feeding the new state through the network
            Q1 = sess.run(Qout, feed_dict = {inputs1: np.identity(16)[s1:s1+1]})

            # Obtain maxQ and set out target value for chosen action
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,action[0]] = reward + y*maxQ1

            # Time to train our network using updated target and predicted Q
            _,W1 = sess.run([updateModel, W], feed_dict = {inputs1: np.identity(16)[s:s+1], nextQ:targetQ})
            rAll += reward
            s = s1
            if done:
                # reduce the chance of random action by updating e
                e = 1./((i/50.) + 10)
                break
        jList.append(j)
        reward_lis.append(rAll)
print "Score over time: " + str(sum(reward_lis) / num_episodes)
