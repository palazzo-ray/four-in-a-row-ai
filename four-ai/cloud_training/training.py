import sys
sys.path.append('../src' )
sys.path.append('../src/agents' )
sys.path.append('../src/envs' )
sys.path.append('../src/utils' )

import numpy as np
import gym
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib


from four_in_a_row_env import FourInARowEnv
from four_dqn_agent import DQNAgent
from four_dqn_agent import DDQNAgent


from stats_logger import StatsLogger


TARGET_NETWORK_UPDATE_FREQUENCY = 2000 
#TARGET_NETWORK_UPDATE_FREQUENCY = 1
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 2000
LOG_FILE_SAVING_FREQUENCY = 1000

class Trainer():
    def __init__(self):
        self.stats_logger = StatsLogger('Training' , '../trained_models/four_a_row/')
        self.total_episode = 0
        self.fit_time = 0

    def training( self, num_round = 1, number_of_episodes=20):

        for r in range(num_round):
            print('=================================================')
            print('round : ' + str(r))
            print('prepare npc')
            
            npc_agent = DDQNAgent( who='npc' , model_name='NN_128x16', load_model=True, save_learnt_to_file=False)
            #npc_agent = None ### random response agent inside the env  
            
            print('preparing agent')
            agent = DDQNAgent( who='player' , model_name='NN_128x16', load_model=True, save_learnt_to_file=True)
            agent.add_fitting_callback(self.fitting_callback)
            
            env = FourInARowEnv(npc_agent=npc_agent)

            self.run_qlearning( r, env, agent, max_number_of_episodes=number_of_episodes)


    def fitting_callback(self, loss,accuracy, q):
        self.fit_time += 1
        self.stats_logger.log_fitting(self.fit_time, loss, accuracy, q)

    def run_qlearning(self, trial_round, env, agent, max_number_of_episodes):

        for episode_number in range(max_number_of_episodes):
                    
            # initialize state
            state = env.reset()
            
            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # repeat for each step of episode, until state is terminal
            while not done:
                
                t += 1 # increase step counter - for display
                
                # choose action from state using policy derived from Q
                action = agent.act(state)
                
                # take action, observe reward and next state
                next_state, reward, done, _ = env.step(action)

                # agent learn (Q-Learning update)
                agent.learn(state, action, reward, next_state, done)
                
                # state <- next state
                state = next_state
                
                R += reward # accumulate reward - for display
                
            
            if episode_number % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0 :
                print('Save model at trial round %s episode : %s' % ( str( trial_round) , str(episode_number) ))
                agent.save_model()

            if episode_number % TARGET_NETWORK_UPDATE_FREQUENCY == 0 :
                print('Update target model at trial round %s episode : %s'  % ( str( trial_round) , str(episode_number) ))
                agent.update_target_network()
        
            self.total_episode += 1
            self.stats_logger.log_iteration(self.total_episode, R, t)

            if self.total_episode % LOG_FILE_SAVING_FREQUENCY == 0:
                self.stats_logger.save_csv()


        print('Save model at trial round %s episode : %s' % ( str( trial_round) , str(episode_number) ))
        agent.save_model()