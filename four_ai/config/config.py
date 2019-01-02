from ..utils.logger import logger

class BaseConfig():
    @staticmethod
    def print_info(x):
        logger.info('Trainer.TARGET_NETWORK_UPDATE_FREQUENCY : ' + str(x.Trainer.TARGET_NETWORK_UPDATE_FREQUENCY ) )
        logger.info('Trainer.MODEL_PERSISTENCE_UPDATE_FREQUENCY : ' + str(x.Trainer.MODEL_PERSISTENCE_UPDATE_FREQUENCY ) )
        logger.info('Trainer.LOG_FILE_SAVING_FREQUENCY : ' + str(x.Trainer.LOG_FILE_SAVING_FREQUENCY ) )
        logger.info('Trainer.NUM_OF_ROUND : ' + str(x.Trainer.NUM_OF_ROUND ) )
        logger.info('Trainer.NUM_OF_ITERATION : ' + str(x.Trainer.NUM_OF_ITERATION ) )

        logger.info('Stats.TRAINING_UPDATE_FREQUENCY  : ' + str(x.Stats.TRAINING_UPDATE_FREQUENCY ) )
        logger.info('Stats.RUN_UPDATE_FREQUENCY  : ' + str(x.Stats.RUN_UPDATE_FREQUENCY ) )
        logger.info('Stats.MAX_LOSS : ' + str(x.Stats.MAX_LOSS) )

        logger.info('Folder.TRAINED_FOLDER: ' + str(x.Folder.TRAINED_FOLDER) )

class Cloud_Config(BaseConfig):

    class Trainer():
        TARGET_NETWORK_UPDATE_FREQUENCY =  20000 
        MODEL_PERSISTENCE_UPDATE_FREQUENCY = 5000 
        LOG_FILE_SAVING_FREQUENCY = 1000

        NUM_OF_ROUND = 100
        NUM_OF_ITERATION = 80000

    class Stats():
        TRAINING_UPDATE_FREQUENCY = 5000 
        RUN_UPDATE_FREQUENCY = 500 
        MAX_LOSS = 1

    class Folder():
        TRAINED_FOLDER = './training_model/'

class Test_Config(BaseConfig):

    class Trainer():
        TARGET_NETWORK_UPDATE_FREQUENCY = 1 
        MODEL_PERSISTENCE_UPDATE_FREQUENCY =1 
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

Config = Test_Config