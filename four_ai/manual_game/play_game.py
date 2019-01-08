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


def play():
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'notebook')

    def showim(buffer):
        ax.imshow(buffer, interpolation='nearest', aspect='auto')
        fig.canvas.draw()

    fig, ax = plt.subplots(figsize=(9, 9))
    npc_agent = DDQNAgent(who='npc', model_name=None, load_model=True, save_learnt_to_file=False)
    npc_agent.epsilon = 0
    npc_agent.epsilon_min = 0
    #npc_agent = None ### random response agent inside the env

    env = FourInARowEnv(npc_agent=npc_agent, test_agent=True)

    env.reset()
    buffer = env.render(mode='rgb_array')
    showim(buffer)

    print("select column 0-6 , or 'q' ")
    while True:
        user_action = input()

        if user_action == 'q':
            break

        state, reward, done, comment = env.step(int(user_action))
        buffer = env.render(mode='rgb_array')
        showim(buffer)

        if done:
            print('Finish : ' + str(done) + '     ' + str(comment))
            break
