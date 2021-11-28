'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

import gym
from problem2 import *

import numpy as np

# set up an environment with fixed random seed for reproducibility
env = gym.make('CartPole-v0')
env.seed(21)

alpha_0 = 0.5
epsilon_0 = 1.0

# set up an agent using tabular Q-learning
agent = Agent_QFunction(env, alpha=alpha_0, epsilon=1.0, gamma=0.95)

# -------------------------------------------------------------------------
def test_epsilon_greedy():
    ''' (10 points) problem2: epsilon_greedy()'''
    obs = env.reset()
    state = agent.encode_state(obs)
    assert agent.epsilon_greedy(state) == 0

    for i in range(10):
        # should generate the same sequence of actions each time the program is run
        action = agent.epsilon_greedy(state)
        if i == 2:
            assert action == 0
        else:
            assert action == 1

# -------------------------------------------------------------------------
def test_learn():
    ''' (20 points) problem2:learn()'''
    obs = env.reset()
    state = agent.encode_state(obs)
    prev_state = state
    prev_action = 0
    prev_reward = 1
    next_state = state
    updated_q_entry = agent.learn(prev_state, prev_action, prev_reward, next_state)
    assert np.allclose(updated_q_entry,
                       np.array([-0.01000946, 0.01946727, 0.0194839, 0.02313647]),
                       atol=0.0001)
    updated_q_entry = agent.learn(prev_state, prev_action, prev_reward, next_state)
    assert np.allclose(updated_q_entry,
                       np.array([-0.01991743, 0.03873715, 0.03877024, 0.04603835]),
                       atol=0.0001)
    updated_q_entry = agent.learn(prev_state, prev_action, prev_reward, next_state)
    assert np.allclose(updated_q_entry,
                       np.array([-0.02972494, 0.05781164, 0.05786103, 0.06870802]),
                       atol=0.0001)
