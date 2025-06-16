import random
import numpy as np
import logging,copy,os,sys
from collections import Counter
import tensorflow as tf
import keras.backend as K

class ValueModel(tf.keras.Model):
    def __init__(self, mode = 0, debug = None, input_dim = [102,2], job_cols=1, window_size = 5, hidden_dim_str = '1000,250', node_module=None, GAMMA=0.99, ALPHA = 0.01, fname='reinforce.h5'): 
        super(ValueModel, self).__init__()
        self.myInfo = "ValueModel"
        self.debug = debug

        self.job_cols = job_cols
        self.window_size = window_size

        self.hidden_dim = []
        self.input_dim = input_dim

        for e in hidden_dim_str.split(','):
            self.hidden_dim.append(int(e))

        print('input_dim',self.input_dim)
        print('hidden_dim',self.hidden_dim)

        self.gamma = float(GAMMA)
        self.lr = ALPHA
        self.G = 0
        self.n_actions = self.window_size

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.action_space = [i for i in range(self.n_actions)]

        self.model_file = fname

        # Define layers
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layer1 = tf.keras.layers.Dense(self.hidden_dim[0],'relu')
        self.hidden_layer2 = tf.keras.layers.Dense(self.hidden_dim[1],'relu')
        self.probs = tf.keras.layers.Dense(self.n_actions, activation='softmax')
        
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        
        # Build the model by calling it with some dummy data
        # self.predict(np.random.rand(1, self.input_dim[0], self.input_dim[1]))


        print('start value model ____________')

    def call(self, inputs):
        flatten = self.flatten(inputs)
        hidden_layer1 = self.hidden_layer1(flatten)
        hidden_layer2 = self.hidden_layer2(hidden_layer1)
        probs = self.probs(hidden_layer2)
        return probs
        
    def choose_action(self, state):
        probabilities = self.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def get_probabilities(self, state):
        probabilities = self.predict(state)[0]
        return probabilities

    def store_transition(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self,state_memory,action_memory,reward_memory):
        state_memory = np.array(state_memory)
        action_memory = np.array(action_memory)
        reward_memory = np.array(reward_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std
        
        with tf.GradientTape() as tape:
            y_pred = self(state_memory, training=True)
            
            # Custom loss calculation
            log_lik = tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1-1e-8))
            log_lik_actions = tf.reduce_sum(log_lik * actions, axis=1)
            loss = -tf.reduce_sum(log_lik_actions * G)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def load_weights(self, filename, lastest_num):
        super().load_weights(filename+"_policy_"+str(lastest_num)+".weights.h5")

    def load_weights_complete_filename(self, policy_filename, predict_filename):
        super().load_weights(policy_filename)

    def save_weights(self, filename, next_num):
        super().save_weights(filename+"_policy_"+str(next_num)+".weights.h5")

    def get_window_size(self):
        return self.window_size
            
            