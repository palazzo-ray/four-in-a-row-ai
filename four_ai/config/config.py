from ..utils.logger import logger


class BaseConfig():
    class Optimizer():
        LEARNING_RATE = 0.001
        LEARNING_RHO = 0.95
        LEARNING_EPSILON = 0.01

        NORMAL_MEMORY_SIZE = 5000
        WINNING_MEMORY_SIZE = 100
        LOSSING_MEMORY_SIZE = 100
        IMPORTANT_MEMORY_SIZE = 2000

        START_TRAINING_SIZE = 80
        BATCH_SIZE = 32
        BATCH_NORMAL = 13  #  41%
        BATCH_WINNING = 5  # 16%
        BATCH_LOSSING = 3  # 9%  - lossing
        BATCH_IMPORTANT = 11  # 34% have more than important num of button on the board

        NUM_BUTTON_PLAYED_AS_IMPORTANT = 15  # 15 button played

    class Explorer():
        GAMMA = 0.95  # DISCOUNT RATE
        EPSILON = 0.4  # EXPLORATION RATE
        EPSILON_MID = 0.2
        EPSILON_DECAY_TO_MID = 0.999997

        EPSILON_MIN = 0.12
        EPSILON_DECAY_TO_MIN = 0.999996

    class Folder():
        TRAINED_FOLDER = './training_model'

    @staticmethod
    def print_info(x):
        logger.info('Trainer.TARGET_NETWORK_UPDATE_FREQUENCY : ' + str(x.Trainer.TARGET_NETWORK_UPDATE_FREQUENCY))
        logger.info('Trainer.MODEL_PERSISTENCE_UPDATE_FREQUENCY : ' + str(x.Trainer.MODEL_PERSISTENCE_UPDATE_FREQUENCY))
        logger.info('Trainer.LOG_FILE_SAVING_FREQUENCY : ' + str(x.Trainer.LOG_FILE_SAVING_FREQUENCY))
        logger.info('Trainer.NUM_OF_ROUND : ' + str(x.Trainer.NUM_OF_ROUND))
        logger.info('Trainer.NUM_OF_ITERATION : ' + str(x.Trainer.NUM_OF_ITERATION))

        logger.info('Stats.TRAINING_UPDATE_FREQUENCY  : ' + str(x.Stats.TRAINING_UPDATE_FREQUENCY))
        logger.info('Stats.RUN_UPDATE_FREQUENCY  : ' + str(x.Stats.RUN_UPDATE_FREQUENCY))
        logger.info('Stats.MAX_LOSS : ' + str(x.Stats.MAX_LOSS))

        logger.info('Optimizer.LEARNING_RATE : ' + str(x.Optimizer.LEARNING_RATE))
        logger.info('Optimizer.LEARNING_RHO : ' + str(x.Optimizer.LEARNING_RHO))
        logger.info('Optimizer.LEARNING_EPSILON : ' + str(x.Optimizer.LEARNING_EPSILON))
        logger.info('Optimizer.NORMAL_MEMORY_SIZE: ' + str(x.Optimizer.NORMAL_MEMORY_SIZE))
        logger.info('Optimizer.WINNING_MEMORY_SIZE: ' + str(x.Optimizer.WINNING_MEMORY_SIZE))
        logger.info('Optimizer.LOSSING_MEMORY_SIZE: ' + str(x.Optimizer.LOSSING_MEMORY_SIZE))
        logger.info('Optimizer.IMPORTANT_MEMORY_SIZE: ' + str(x.Optimizer.IMPORTANT_MEMORY_SIZE))

        logger.info('Optimizer.BATCH_SIZE : ' + str(x.Optimizer.BATCH_SIZE))
        logger.info('Optimizer.BATCH_NORMAL : ' + str(x.Optimizer.BATCH_NORMAL))
        logger.info('Optimizer.BATCH_WINNING : ' + str(x.Optimizer.BATCH_WINNING))
        logger.info('Optimizer.BATCH_LOSSING : ' + str(x.Optimizer.BATCH_LOSSING))
        logger.info('Optimizer.BATCH_IMPORTANT: ' + str(x.Optimizer.BATCH_IMPORTANT))

        logger.info('Explorer.GAMMA : ' + str(x.Explorer.GAMMA))
        logger.info('Explorer.EPSILON : ' + str(x.Explorer.EPSILON))
        logger.info('Explorer.EPSILON_MID : ' + str(x.Explorer.EPSILON_MID))
        logger.info('Explorer.EPSILON_DECAY_TO_MID : ' + str(x.Explorer.EPSILON_DECAY_TO_MID))
        logger.info('Explorer.EPSILON_MIN : ' + str(x.Explorer.EPSILON_MIN))
        logger.info('Explorer.EPSILON_DECAY_TO_MIN : ' + str(x.Explorer.EPSILON_DECAY_TO_MIN))

        logger.info('Folder.TRAINED_FOLDER: ' + str(x.Folder.TRAINED_FOLDER))


class Cloud_Config(BaseConfig):
    class Trainer():
        TARGET_NETWORK_UPDATE_FREQUENCY = 20000
        MODEL_PERSISTENCE_UPDATE_FREQUENCY = 5000
        LOG_FILE_SAVING_FREQUENCY = 1000

        NUM_OF_ROUND = 10000
        NUM_OF_ITERATION = 200000

    class Stats():
        TRAINING_UPDATE_FREQUENCY = 100
        RUN_UPDATE_FREQUENCY = 100
        MAX_LOSS = 1


class Test_Config(BaseConfig):
    class Trainer():
        TARGET_NETWORK_UPDATE_FREQUENCY = 1
        MODEL_PERSISTENCE_UPDATE_FREQUENCY = 1
        LOG_FILE_SAVING_FREQUENCY = 1

        NUM_OF_ROUND = 1
        NUM_OF_ITERATION = 20

    class Stats():
        #TRAINING_UPDATE_FREQUENCY = 1000
        #RUN_UPDATE_FREQUENCY = 10
        TRAINING_UPDATE_FREQUENCY = 1
        RUN_UPDATE_FREQUENCY = 1
        MAX_LOSS = 1


#Config = Test_Config
Config = Cloud_Config
