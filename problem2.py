# -------------------------------------------------------------------------
'''
    Problem 2: Implement Q-learning using function approximation.
'''

import random
import numpy as np

class Agent_QFunction(object):
    '''' Agent that learn via function approximation Q-learning. '''

    #--------------------------
    def __init__(self, env, alpha=0.1, epsilon=1.0, gamma=0.9, decay=0.01):
        """
        :param env: Environment defined by Gym
        :param alpha: learning rate for updating the Q-table
        :param epsilon: Epsilon in epsilon-greedy policy
        :param gamma: discounting factor
        :param decay: weight decay parameter when updating the policy parameter self.w
        """
        np.random.seed(21)
        random.seed(21)

        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.w = np.zeros((4, 2))   # parameter of two linear regression models, each column for an action.
        self.decay = decay          # adding l-2 regularization when updating self.w

        # Following variables for statistics
        self.training_trials = 0
        self.testing_trials = 0

    #--------------------------
    def encode_state(self, observed_state):
        return np.array(observed_state)

    #--------------------------
    def epsilon_greedy(self, state):
        '''
        Given a state (represented by an 1x4 array), choose an action using epsilon-greedy policy
                with the current Q function.
        :param state: 1x4 array
        :return action that agent will take. Should be either 0 or 1
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Q0 = np.dot(self.w.T[0], state)
        Q1 = np.dot(self.w.T[1], state)
        if Q0 > Q1:
            return 0
        elif Q0 < Q1:
            return 1
        else:
            return self.env.action_space.sample()
        #########################################

    #--------------------------
    def learn(self, prev_state, prev_action, prev_reward, next_state):
        '''
        Update Q[prev_state, prev_action] using the Q-learning formula.

        :param prev_state: previous state vector.
        :param prev_action: action taken at the previous state.
        :param prev_reward: reward when taking the previous action.
        :param next_state: state vector after taking the previous action
        :return return the updated q entry
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        w = self.w.T[prev_action].T
        w_increased = w + self.decay
        q_increased = np.dot(w_increased.T, state)
        q = np.dot(w.T, state)

        dq = q_increased - q
        gradient = dq / self.decay
        
        q_new = np.dot(w.T, next_state)
        q_old = np.dot(w.T, prev_state)

        w_tmp = self.w.T

        update_value = w_tmp[prev_action] + \
            self.alpha * gradient * (self.gamma * q_new - q_old + prev_reward)
        
        self.w[prev_action]  =  update_value

        return update_value


        #########################################