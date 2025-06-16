import random
import numpy as np
import logging, copy, os, sys
from collections import Counter
import tensorflow as tf
import keras.backend as K


class ValueModel(tf.keras.Model):
    def __init__(
        self,
        mode=0,
        debug=None,
        input_dim=[102, 2],
        job_cols=1,
        window_size=5,
        hidden_dim_str="1000,250",
        node_module=None,
        GAMMA=0.99,
        ALPHA=0.01,
        fname="reinforce.h5",
        algorithm="pg",
    ):
        super(ValueModel, self).__init__()
        self.myInfo = "ValueModel"
        self.debug = debug
        self.algorithm = algorithm

        self.job_cols = job_cols
        self.window_size = window_size

        self.hidden_dim = []
        self.input_dim = input_dim

        for e in hidden_dim_str.split(","):
            self.hidden_dim.append(int(e))

        print("input_dim", self.input_dim)
        print("hidden_dim", self.hidden_dim)

        self.gamma = float(GAMMA)
        self.lr = ALPHA
        self.G = 0
        self.n_actions = self.window_size

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.action_space = [i for i in range(self.n_actions)]

        self.model_file = fname

        # Define layers following the DRAS paper architecture
        # First reshape the 2-D input so we can apply a 2-D convolution
        self.reshape = tf.keras.layers.Reshape(
            (self.input_dim[0], self.input_dim[1], 1)
        )
        # 1x2 convolution with a single filter.  This yields one value per input
        # row, effectively combining the two features of each element.
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, self.input_dim[1]),
            padding="valid",
        )
        self.conv_act = tf.keras.layers.LeakyReLU()

        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layer1 = tf.keras.layers.Dense(self.hidden_dim[0])
        self.hidden_act1 = tf.keras.layers.LeakyReLU()
        self.hidden_layer2 = tf.keras.layers.Dense(self.hidden_dim[1])
        self.hidden_act2 = tf.keras.layers.LeakyReLU()
        self.probs = tf.keras.layers.Dense(self.n_actions, activation="softmax")
        if self.algorithm == "ppo":
            self.value_head = tf.keras.layers.Dense(1)

        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

        # Build the model so weights can be loaded immediately
        self.build((None, self.input_dim[0], self.input_dim[1]))

        print("start value model ____________")

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv(x)
        x = self.conv_act(x)
        x = self.flatten(x)
        x = self.hidden_layer1(x)
        x = self.hidden_act1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_act2(x)
        probs = self.probs(x)
        if self.algorithm == "ppo":
            value = self.value_head(x)
            return probs, value
        return probs

    def choose_action(self, state, return_prob=False):
        state = np.asarray(state)
        if self.algorithm == "ppo":
            probabilities, _ = self(state, training=False)
            probabilities = probabilities.numpy()[0]
        else:
            probabilities = self(state, training=False).numpy()[0]
        action = np.random.choice(self.action_space, p=probabilities)
        if return_prob:
            return action, probabilities[action]
        return action

    def get_probabilities(self, state):
        state = np.asarray(state)
        if self.algorithm == "ppo":
            probabilities, _ = self(state, training=False)
            return probabilities.numpy()[0]
        probabilities = self(state, training=False).numpy()[0]
        return probabilities

    def store_transition(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self, state_memory, action_memory, reward_memory, old_action_probs=None):
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
        # Scale the normalised returns so the gradients are large enough to
        # dominate the Adam moment estimates as suggested in the DRAS paper.
        G *= 100.0

        if self.algorithm == "ppo":
            return self._learn_ppo(state_memory, actions, G, old_action_probs)

        with tf.GradientTape() as tape:
            y_pred = self(state_memory, training=True)

            # Custom loss calculation
            log_lik = tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8))
            log_lik_actions = tf.reduce_sum(log_lik * actions, axis=1)
            loss = -tf.reduce_sum(log_lik_actions * G)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def _learn_ppo(self, states, actions_onehot, returns, old_action_probs):
        if old_action_probs is None:
            raise ValueError("old_action_probs required for PPO")

        with tf.GradientTape() as tape:
            probs, values = self(states, training=True)
            values = tf.squeeze(values, axis=1)
            log_probs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
            log_probs_act = tf.reduce_sum(log_probs * actions_onehot, axis=1)
            ratios = tf.exp(log_probs_act - tf.math.log(old_action_probs + 1e-8))
            advantages = returns - values
            clip_eps = 0.2
            surrogate1 = ratios * advantages
            surrogate2 = (
                tf.clip_by_value(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
            )
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            value_loss = tf.reduce_mean(tf.square(advantages))
            loss = policy_loss + 0.5 * value_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def load_weights(self, filename, lastest_num):
        if not self.built:
            self.build((None, self.input_dim[0], self.input_dim[1]))
        super().load_weights(filename + "_policy_" + str(lastest_num) + ".weights.h5")

    def load_weights_complete_filename(self, policy_filename, predict_filename):
        if not self.built:
            self.build((None, self.input_dim[0], self.input_dim[1]))
        super().load_weights(policy_filename)

    def save_weights(self, filename, next_num):
        super().save_weights(filename + "_policy_" + str(next_num) + ".weights.h5")

    def get_window_size(self):
        return self.window_size
