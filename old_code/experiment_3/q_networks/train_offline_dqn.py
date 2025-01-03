import os 
import sys
import gym
import pickle
import random
import utils 

import numpy as np

from collections import deque

from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from matplotlib import pyplot as plt 

MEMORY_PATH = './buffers/CartPole-v0/1/DQN/'

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.00001
        self.batch_size = 256
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.get_memory_buffer()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_memory_buffer(self):
        memory_buffer_path = os.path.join(MEMORY_PATH, 'memory_buffer.p')
        
        with open(memory_buffer_path, 'rb') as f:
            self.memory = pickle.load(f)

        print(len(self.memory))

    def get_action(self, state):
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    def train_model(self): 
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size,
					   epochs=1, verbose=0)

def run_DQN():
    episodes = 500 
    seed = 2 
    results = []
    game = 'CartPole-v0'

    env = gym.make(game)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.train_model()

            score += reward
            state = next_state

        print(score)
        results.append(score)
    
    utils.save_offline_results(game, 'DQN', seed, results)

    plt.plot(results)
    plt.show()

run_DQN()
