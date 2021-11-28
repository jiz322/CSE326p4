# -------------------------------------------------------------------------
'''
    Problem 3: Implement the Q-learning algorithm (on-policy control) for both the tabular and functional agent.
'''

import gym
import numpy as np
import argparse
import pickle

from problem1 import Agent_QTable
from problem2 import Agent_QFunction


# --------------------------
def environment_info(env):
    ''' Prints info about the given environment. '''
    print('************** Environment Info **************')
    print('Observation space: {}'.format(env.observation_space))
    print('Observation space high values: {}'.format(env.observation_space.high))
    print('Observation space low values: {}'.format(env.observation_space.low))
    print('Action space: {}'.format(env.action_space))
    print()

# --------------------------
def q_learning(env, agent, alpha_0=0.9, epsilon_0=1):
    '''
    This function has 20 points.
    Implement the Q-learning algorithm. This should work for both types of agents.
    Refer to the book 'An Introduction to Reinforcement Learning, 2018 edition', Section 6.5 for the Q-learning algorithm.
    :param env: Gym enviroment object.
    :param agent: Learning agent.
    :param alpha_0: initial learning rate.
    :param epsilon_0: initial exploration rate.
    :return Rewards for training/testing and epsilon/alpha value history.
    '''
    # collect data for tracking the performance
    training_totals = []
    testing_totals = []
    history = {'epsilon': [], 'alpha': []}

    # start training: run 2000 episodes and train the policy (Q-function).
    for episode in range(2000):
    	# generate 2000 episodes for training. 
        # agent will learn from these data using Q-learning.
        if episode % 20 == 0:
            print(f'episode={episode}')
        episode_rewards = 0
        obs = env.reset()

        # Decay epsilon as training goes on. Minimum rate = 0.01
        # this decay can be too fast to lose exploration and need to be tuned carefully.
        # This setting is good for tabular Q-learning
        agent.epsilon = max(0.01, epsilon_0 * np.exp(-0.005 * episode))
        history['epsilon'].append(agent.epsilon)

        # Decay learning rate as well. Minimum rate = 0.1
        # This setting is good for tabular Q-learning
        agent.alpha = max(0.01, alpha_0 * np.exp(-0.005 * episode))
        history['alpha'].append(agent.alpha)

        for step in range(200):        # 200 steps max
            # Each episode runs for a maximal of 200 steps.
        	#########################################
        	## INSERT YOUR CODE HERE
        	#########################################

        # record the total reward of this episode
        training_totals.append(episode_rewards)
        agent.training_trials += 1


    # start testing: run 100 episodes and apply the learned policy (Q-function) to act on the states.
    agent.epsilon = 0   # only greedy selection and no exploration
    for episode in range(100):
        env.reset()
        episode_rewards = 0
        for step in range(200):        # 200 steps max
            # Each episode runs for a maximal of 200 steps.
        	#########################################
        	## INSERT YOUR CODE HERE
        	#########################################

        # record the total reward of this episode
        testing_totals.append(episode_rewards)
        agent.testing_trials += 1

    return training_totals, testing_totals, history


# --------------------------
def main():
    ''' Execute main program. '''
    # Create a cartpole environment
    # Observation: [horizontal pos, velocity, angle of pole, angular velocity]
    # Rewards: +1 at every step. i.e. goal is to stay alive
    env = gym.make('CartPole-v0')
    # Set environment seed
    env.seed(21)
    environment_info(env)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', help='define type of agent you want')
    args = parser.parse_args()
    alpha_0 = 0.5
    epsilon_0 = 1.0
    decay = 0.01

    # Q-learning
    if args.agent == 'tabular':
        agent = Agent_QTable(env, alpha=alpha_0, epsilon=epsilon_0, gamma=0.99)
    else:
        agent = Agent_QFunction(env, alpha=alpha_0, epsilon=epsilon_0, gamma=0.99, decay=decay)

    training_totals, testing_totals, history = q_learning(env, agent, alpha_0, epsilon_0)

    # stats.display_stats(agent, training_totals, testing_totals, history)
    # stats.save_info(agent, training_totals, testing_totals)

    # Check if environment is solved
    # if np.mean(testing_totals) >= 195.0:
    #     print("Environment SOLVED!!!")
    # else:
    #     print("Environment not solved.",
    #           "Must get average reward of 195.0 or",
    #           "greater for 100 consecutive trials.")

    # save the results to a pickle file that will be visualized
    with open(f'../data/{args.agent}.pkl', 'wb') as out_f:
        pickle.dump({'agent': agent,
                     'agent_type': args.agent,
                     'training rewards': training_totals,
                     'test rewards': testing_totals,
                     'parameter history': history},
                    out_f)

if __name__ == '__main__':
    ''' Run main program. '''
    main()