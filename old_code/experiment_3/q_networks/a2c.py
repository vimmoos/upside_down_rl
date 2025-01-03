import os 
import sys
import gym
import utils 

import numpy as np

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from matplotlib import pyplot as plt 

class A2CAgent:
	def __init__(self, state_size, action_size):
		self.render = False
		self.state_size = state_size
		self.action_size = action_size
		self.value_size = 1
		self.discount_factor = 0.99
		self.actor_lr = 0.001
		self.critic_lr = 0.005
		self.actor = self.build_actor()
		self.critic = self.build_critic()

	def build_actor(self):
		actor = Sequential()
		actor.add(Dense(24, input_dim=self.state_size, activation='relu',
						kernel_initializer='he_uniform'))
		actor.add(Dense(self.action_size, activation='softmax',
						kernel_initializer='he_uniform'))
		actor.compile(loss='categorical_crossentropy',
					  optimizer=Adam(lr=self.actor_lr))
		
		return actor

	def build_critic(self):
		critic = Sequential()
		critic.add(Dense(24, input_dim=self.state_size, activation='relu',
						 kernel_initializer='he_uniform'))
		critic.add(Dense(self.value_size, activation='linear',
						 kernel_initializer='he_uniform'))
		critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
		
		return critic

	def get_action(self, state):
		policy = self.actor.predict(state, batch_size=1).flatten()
			
		return np.random.choice(self.action_size, 1, p=policy)[0]

	def train_model(self, state, action, reward, next_state, done):
		target = np.zeros((1, self.value_size))
		advantages = np.zeros((1, self.action_size))

		value = self.critic.predict(state)[0]
		next_value = self.critic.predict(next_state)[0]

		if done:
			advantages[0][action] = reward - value
			target[0][0] = reward
		else:
			advantages[0][action] = reward + self.discount_factor * (next_value) - value
			target[0][0] = reward + self.discount_factor * next_value

		self.actor.fit(state, advantages, epochs=1, verbose=0)
		self.critic.fit(state, target, epochs=1, verbose=0)

def run_A2C():
    episodes = 500
    seed = 1
    results = []
    game = 'CartPole-v0'

    env = gym.make(game)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size)

    for e in range(episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

        results.append(score)

    utils.save_trained_model(game, seed, 'A2C', agent.actor)

    plt.plot(results)
    plt.show()

run_A2C()
