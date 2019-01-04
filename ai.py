import logging
from four_ai.cloud_training import training
from four_ai.manual_game import play_game
from four_ai.utils.logger import logger
from four_ai.config.config import Config


def play():
    play_game.play()


if __name__ == "__main__":
    logger.info('==========================================================================')
    logger.info('|                                                                        |')
    logger.info('|                                                                        |')
    logger.info('|         Start training                                                 |')
    logger.info('|                                                                        |')
    logger.info('|                                                                        |')
    logger.info('==========================================================================')

    Config.print_info(Config)

    logger.info('')
    logger.info('')

    trainer = training.Trainer()
    #trainer.training()
    trainer.training(num_round=Config.Trainer.NUM_OF_ROUND, number_of_episodes=Config.Trainer.NUM_OF_ITERATION)
