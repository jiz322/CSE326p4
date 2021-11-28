'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

import gym
from problem1 import *

import numpy as np

# set up an environment with fixed random seed for reproducibility
env = gym.make('CartPole-v0')

alpha_0 = 0.5
epsilon_0 = 1.0

# set up an agent using tabular Q-learning
agent = Agent_QTable(env, alpha=alpha_0, epsilon=1.0, gamma=0.95)

# -------------------------------------------------------------------------
def test_encode_state():
    ''' (10 points) encode_state()'''
    env.seed(21)
    obs = env.reset()
    state = agent.encode_state(obs)
    assert state == 4555

# -------------------------------------------------------------------------
def test_epsilon_greedy():
    ''' (10 points) problem1: epsilon_greedy()'''
    env.seed(21)
    obs = env.reset()
    state = agent.encode_state(obs)
    action = agent.epsilon_greedy(state)
    assert action == 0
    agent.Q_table[state] = [0, 1]
    for i in range(10):
        # should generate the same sequence of actions each time the program is run
        action = agent.epsilon_greedy(state)
        if i == 2:
            assert action == 0
        else:
            assert action == 1

# -------------------------------------------------------------------------
def test_learn():
    ''' (20 points) problem1:learn()'''
    obs = env.reset()
    state = agent.encode_state(obs)
    prev_state = state
    prev_action = 0
    prev_reward = 1
    next_state = state
    agent.Q_table[state] = [0, 1]
    updated_q_entry = agent.learn(prev_state, prev_action, prev_reward, next_state)
    assert updated_q_entry == 0.975
    updated_q_entry = agent.learn(prev_state, prev_action, prev_reward, next_state)
    assert updated_q_entry == 1.4625
    updated_q_entry = agent.learn(prev_state, prev_action, prev_reward, next_state)
    assert np.allclose(updated_q_entry,1.9259, atol=0.0001)
