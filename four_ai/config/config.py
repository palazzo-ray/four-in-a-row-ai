
class Config():

    class Trainer():
        TARGET_NETWORK_UPDATE_FREQUENCY =  20000 
        MODEL_PERSISTENCE_UPDATE_FREQUENCY = 5000 
        LOG_FILE_SAVING_FREQUENCY = 1

    class Stats():
        #TRAINING_UPDATE_FREQUENCY = 1000
        #RUN_UPDATE_FREQUENCY = 10
        TRAINING_UPDATE_FREQUENCY = 1 
        RUN_UPDATE_FREQUENCY = 1 
        MAX_LOSS = 1

    class Folder():
        TRAINED_FOLDER = './training_model/'