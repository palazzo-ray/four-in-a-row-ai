import sys

import numpy as np
import gym
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib

from ..envs.four_in_a_row_env import FourInARowEnv
from ..agents.four_dqn_agent import DQNAgent
from ..agents.four_dqn_agent import DDQNAgent

from ..utils.stats_logger import StatsLogger
from ..utils.logger import logger

from ..config.config import Config


class Trainer():
    def __init__(self):
        self.stats_logger = StatsLogger('Training', Config.Folder.TRAINED_FOLDER)
        self.total_episode = 0
        self.fit_time = 0
        self.winning = 0
        self.lossing = 0
        self.draw_game = 0
        self.wrong_move = 0

        self.round_win = 0

    def get_npc(self):
        logger.info('prepare npc')
        npc_agent = DDQNAgent(who='npc', model_name=None, load_model=True, save_learnt_to_file=False)
        npc_agent.epsilon = 0.3

        return npc_agent

    def training(self, num_round=1, number_of_episodes=20):

        round_winning_rate = 0
        npc_agent = self.get_npc()

        logger.info('preparing agent')
        agent = DDQNAgent(who='player', model_name=None, load_model=True, save_learnt_to_file=True)
        self.agent = agent
        agent.add_fitting_callback(self.fitting_callback)

        for r in range(num_round):
            logger.info('=================================================')
            logger.info('round : ' + str(r))


            if round_winning_rate > 0.51:
                logger.info('upgrade npc agent')
                npc_agent = self.get_npc()

            env = FourInARowEnv(npc_agent=npc_agent)

            self.run_qlearning(r, env, agent, max_number_of_episodes=number_of_episodes)

            logger.info('')
            logger.info('saving model backup')
            agent.save_model(backup_copy_name='agent_round_' + str(r))
            npc_agent.save_model(backup_copy_name='npc_round_' + str(r))

            round_winning_rate = (self.round_win / num_round)
            logger.info('Round winning rate : ' + str(number_of_episodes))
            self.round_win = 0

    def fitting_callback(self, loss, accuracy, q):
        self.fit_time += 1
        self.stats_logger.log_fitting(self.fit_time, loss, accuracy, q, self.agent.epsilon)

    def _update_game_result(self, scn):
        if scn == 'player_win':
            self.winning += 1
            self.round_win += 1
        elif scn == 'npc_win':
            self.lossing += 1
        elif scn == 'player_wrong_move':
            self.wrong_move += 1
        elif scn == 'no_space':
            self.draw_game += 1

    def run_qlearning(self, trial_round, env, agent, max_number_of_episodes):

        for episode_number in range(max_number_of_episodes):

            # initialize state
            state = env.reset()

            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode i.e episode length
            scn = ''

            # repeat for each step of episode, until state is terminal
            while not done:

                t += 1  # increase step counter - for display

                # choose action from state using policy derived from Q
                action = agent.act(state)

                # take action, observe reward and next state
                next_state, reward, done, scn = env.step(action)

                # agent learn (Q-Learning update)
                agent.learn(state, action, reward, next_state, done, scn)

                # state <- next state
                state = next_state

                R += reward  # accumulate reward - for display

            if self.total_episode % Config.Trainer.MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
                logger.info('Save model at trial round %s episode : %s' % (str(trial_round), str(episode_number)))
                agent.save_model()

            if self.total_episode % Config.Trainer.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                logger.info(
                    'Update target model at trial round %s episode : %s' % (str(trial_round), str(episode_number)))
                agent.update_target_network()

            self.total_episode += 1

            self._update_game_result(scn)
            self.stats_logger.log_iteration(self.total_episode, R, t, self.winning, self.lossing, self.draw_game,
                                            self.wrong_move)

            if self.total_episode % Config.Trainer.LOG_FILE_SAVING_FREQUENCY == 0:
                self.stats_logger.save_csv()

        logger.info('Save model at trial round %s episode : %s' % (str(trial_round), str(episode_number)))
        agent.save_model()