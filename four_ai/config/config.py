from ..utils.logger import logger


class BaseConfig():
    class Optimizer():
        LEARNING_RATE = 0.00025
        LEARNING_RHO = 0.95
        LEARNING_EPSILON = 0.01

        MEMORY_SIZE = 20000

    class Explorer():
        GAMMA = 0.95  # DISCOUNT RATE
        EPSILON = 0.3  # EXPLORATION RATE
        EPSILON_MIN = 0.001
        EPSILON_DECAY = 0.999995

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

        logger.info('Optimizer.LEARNING_RATE : ' + str(x.Optimizer.LEARNING_RATE ) )
        logger.info('Optimizer.LEARNING_RHO : ' + str(x.Optimizer.LEARNING_RHO ) )
        logger.info('Optimizer.LEARNING_EPSILON : ' + str(x.Optimizer.LEARNING_EPSILON) )
        logger.info('Optimizer.MEMORY_SIZE: ' + str(x.Optimizer.MEMORY_SIZE) )

        logger.info('Explorer.GAMMA : ' + str( x.Explorer.GAMMA ) )
        logger.info('Explorer.EPSILON : ' + str( x.Explorer.EPSILON ) )
        logger.info('Explorer.EPSILON_MIN : ' + str( x.Explorer.EPSILON_MIN ) )
        logger.info('Explorer.EPSILON_DECAY : ' + str( x.Explorer.EPSILON_DECAY ) )

        logger.info('Folder.TRAINED_FOLDER: ' + str(x.Folder.TRAINED_FOLDER))


class Cloud_Config(BaseConfig):
    class Trainer():
        TARGET_NETWORK_UPDATE_FREQUENCY = 7500
        MODEL_PERSISTENCE_UPDATE_FREQUENCY = 5000
        LOG_FILE_SAVING_FREQUENCY = 1000

        NUM_OF_ROUND = 10000
        NUM_OF_ITERATION = 25000


    class Stats():
        TRAINING_UPDATE_FREQUENCY = 500
        RUN_UPDATE_FREQUENCY = 500
        MAX_LOSS = 1

    class Folder():
        TRAINED_FOLDER = './training_model/'


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

    class Folder():
        TRAINED_FOLDER = './training_model/'


#Config = Test_Config
Config = Cloud_Config
