import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

import numpy as np

from actor import Actor
from critic import Critic
from replaybuffer import RBuffer


class Agent:
    def __init__(self, env, alg, n_layers_actor, n_units_actor, n_layers_critic, n_units_critic):
        n_action = len(env.action_space.high)

        # initialize actor and critic main and target networks
        self.actor_main = Actor(n_action, n_layers_actor, n_units_actor)
        self.actor_target = Actor(n_action, n_layers_actor, n_units_actor)
        self.a_opt = tf.keras.optimizers.Adam(0.0005)
        self.actor_target.compile(optimizer=self.a_opt)

        self.critic_main = Critic(n_layers_critic, n_units_critic)
        self.critic_target = Critic(n_layers_critic, n_units_critic)
        self.c_opt1 = tf.keras.optimizers.Adam(0.0005)
        self.critic_target.compile(optimizer=self.c_opt1)

        # initialize extra network for TD3 and TDSPG
        self.critic_main2 = Critic(n_layers_critic, n_units_critic)
        self.critic_target2 = Critic(n_layers_critic, n_units_critic)
        self.c_opt2 = tf.keras.optimizers.Adam(0.0005)
        self.critic_target2.compile(optimizer=self.c_opt2)

        self.tau = 0.001

        self.batch_size = 32
        self.n_actions = len(env.action_space.high)

        # initialize experience replay buffer with a maximum size of 10000
        self.memory = RBuffer(10000, env.observation_space.shape, len(env.action_space.high))

        self.trainstep = 0

        self.gamma = 0.99

        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]

        self.warmup = 200

        self.alg = alg
        self.actor_update_steps = 2
        if self.alg == 'spg' or self.alg == 'dpg':
            self.actor_update_steps = 1

    def update_target(self, tau=None):

        if tau is None:
            tau = self.tau

        weights1 = []
        targets1 = self.actor_target.weights
        for i, weight in enumerate(self.actor_main.weights):
            weights1.append(weight * tau + targets1[i] * (1 - tau))
        self.actor_target.set_weights(weights1)

        weights2 = []
        targets2 = self.critic_target.weights
        for i, weight in enumerate(self.critic_main.weights):
            weights2.append(weight * tau + targets2[i] * (1 - tau))
        self.critic_target.set_weights(weights2)

    def act(self, state):
        """perform feed forward step through the actor network and return its output as the selected action"""

        # use state to retrieve action
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor_main(state)

        # for the first 'warmup' trainsteps, add some noise to the action
        if self.trainstep <= self.warmup:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)
        actions = self.max_action * (tf.clip_by_value(actions, self.min_action, self.max_action))

        # convert action from tensor to numpy array
        return actions[0].numpy()

    def train(self, sigma):
        """Train the actor and critic networks using samples from the experience replay buffer,
        Code inspired by:
        https://towardsdatascience.com/deep-deterministic-and-twin-delayed-deep-deterministic-policy-gradient-with-tensorflow-2-x-43517b0e0185
        """

        # only start training if the buffer is filled to a certain amount
        if self.memory.cnt < (10 * self.batch_size):
            return 0, 0

        # sample tuple from memory, convert to tensors
        states_raw, actions_raw, rewards_raw, next_states_raw, deads_raw = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states_raw, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states_raw, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards_raw, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions_raw, dtype=tf.float32)

        # Update Critic(s):
        if self.alg == 'spg' or self.alg == 'dpg':
            # for spg and dpg use a single critic (+target critic) to update according to the bellman equation
            with tf.GradientTape() as tape:

                target_actions = self.actor_target(next_states)
                target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 1)

                critic_value = tf.squeeze(self.critic_main(states, actions), 1)
                target_values = rewards + self.gamma * target_next_state_values * np.invert(deads_raw)

                critic_loss = tf.keras.losses.MSE(target_values, critic_value)

            grads2 = tape.gradient(critic_loss, self.critic_main.trainable_variables)

            self.c_opt1.apply_gradients(zip(grads2, self.critic_main.trainable_variables))

        elif self.alg == 'td3' or self.alg == 'tdspg':
            # td3 and tdspg make use of a pair of critics (+targets)
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

                # retrieve next state's actions and apply noise
                target_actions = self.actor_target(next_states)
                target_actions += tf.clip_by_value(tf.random.normal(shape=[*np.shape(target_actions)], mean=0.0,
                                                                    stddev=0.2), -0.5, 0.5)
                target_actions = self.max_action * (tf.clip_by_value(target_actions, self.min_action, self.max_action))

                # compute the next state values for both critic networks
                target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 1)
                critic_value = tf.squeeze(self.critic_main(states, actions), 1)

                target_next_state_values2 = tf.squeeze(self.critic_target2(next_states, target_actions), 1)
                critic_value2 = tf.squeeze(self.critic_main2(states, actions), 1)

                # select the lowest next state value
                next_state_target_value = tf.math.minimum(target_next_state_values, target_next_state_values2)

                # apply bellman equation
                target_values = rewards + self.gamma * next_state_target_value * np.invert(deads_raw)

                # calculate losses for both networks making use of te same target value
                critic_loss1 = tf.keras.losses.MSE(target_values, critic_value)
                critic_loss2 = tf.keras.losses.MSE(target_values, critic_value2)

            grads1 = tape1.gradient(critic_loss1, self.critic_main.trainable_variables)
            grads2 = tape2.gradient(critic_loss2, self.critic_main2.trainable_variables)

            self.c_opt1.apply_gradients(zip(grads1, self.critic_main.trainable_variables))
            self.c_opt2.apply_gradients(zip(grads2, self.critic_main2.trainable_variables))

        # Update Actor:
        self.trainstep += 1
        if self.trainstep % self.actor_update_steps == 0:
            # the actor for td3 and tdspg only updates every other epoch
            if self.alg == 'td3' or self.alg == 'dpg':
                with tf.GradientTape() as tape3:
                    # update actor according to the gradients through the critic network

                    # calculate actions according to the current policy
                    new_policy_actions = self.actor_main(states)

                    # calculate the loss according to the critic given the current policy
                    actor_loss = -self.critic_main(states, new_policy_actions)
                    actor_loss = tf.math.reduce_mean(actor_loss)
            elif self.alg == 'tdspg' or self.alg == 'spg':
                with tf.GradientTape() as tape3:
                    # calculate actions according to the current policy
                    new_policy_actions = self.actor_main(states)
                    best = new_policy_actions

                    # calculate the current value of the policy
                    QcurrentPolicy = self.critic_main(states, new_policy_actions)

                    # find where the taken actions
                    idxGreater = np.where(self.critic_main(states, actions) > QcurrentPolicy)[0]

                    best_raw = best.numpy()
                    best_raw[idxGreater] = actions_raw[idxGreater]
                    best = tf.convert_to_tensor(best_raw, dtype=tf.float32)

                    n_sampled_actions = 5
                    for _ in range(n_sampled_actions):
                        best_raw = best.numpy()

                        # sample around current best action, clip between -1 and 1
                        sampled_raw = best_raw + np.random.normal(0, sigma, size=np.shape(best_raw))
                        sampled_raw = sampled_raw.clip(-1, 1)
                        sampled = tf.convert_to_tensor(sampled_raw, dtype=tf.float32)

                        # check whether new sampled action has a higher critic value than current best action
                        idxGreater = np.where(self.critic_main(states, sampled) > self.critic_main(states, best))[0]

                        # update best actions
                        best_raw[idxGreater] = sampled_raw[idxGreater]
                        best = tf.convert_to_tensor(best_raw, dtype=tf.float32)

                    target = best

                    # actor loss is the mean squared error between selected action and sampled best action
                    actor_loss = tf.reduce_mean(tf.squared_difference(new_policy_actions, target))

            grads3 = tape3.gradient(actor_loss, self.actor_main.trainable_variables)
            self.a_opt.apply_gradients(zip(grads3, self.actor_main.trainable_variables))

        self.update_target()

        # return value loss and policy loss
        if self.alg == 'spg' or self.alg == 'dpg':
            av_v_loss = np.sum(critic_loss.numpy()) / critic_loss.numpy().size
            av_p_loss = np.sum(actor_loss.numpy()) / actor_loss.numpy().size
        else:
            av_v_loss = (np.sum(critic_loss1.numpy()) + np.sum(critic_loss2.numpy())) / (critic_loss1.numpy().size + critic_loss2.numpy().size)
            try:
                av_p_loss = np.sum(actor_loss.numpy()) / actor_loss.numpy().size
            except:
                return av_v_loss, 0

        return av_v_loss, av_p_loss

    def savexp(self, state, next_state, action, dead, reward):
        self.memory.store_transition(state, action, reward, next_state, dead)
