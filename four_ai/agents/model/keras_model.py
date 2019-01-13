
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import keras.models as km
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, Activation, LeakyReLU
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from ...config.config import Config
from ...utils.logger import logger
import os.path
import numpy as np


class BaseModel():
    def __init__(self, model_name, my_button, opponent_button, model_save_path):
        self.model_save_path = model_save_path
        self.model_name = model_name

        self.my_button = my_button
        self.opponent_button = opponent_button
        self.fit_count = 0

    def save_model_backup_copy(self, backup_name):
        file_name = self.model_save_path + '/' + self.model_name

        model = self.model

        full_model_file = file_name + '_' + backup_name + '_model.dat'

        model.save(full_model_file)

        logger.info("Saved backup model to disk : " + str(full_model_file))

    def save_model(self):
        file_name = self.model_save_path + '/' + self.model_name

        model = self.model

        full_model_file = file_name + '_full_model.dat'

        model.save(full_model_file)

        logger.info("Saved model to disk : " + str(file_name))

    def load_model_from_file(self):
        file_name = self.model_save_path + '/' + self.model_name
        full_model_file = file_name + '_full_model.dat'

        # check exist
        abs_path = os.path.abspath(str(full_model_file))
        if not os.path.isfile(full_model_file):
            logger.info('Model file not exist: ' + str(full_model_file))
            logger.info('                    : ' + str(abs_path))
            return False

        self.model = km.load_model(full_model_file)

        logger.info("Loaded model from disk : " + str(file_name))

        return True

    def predict(self, state):
        act_values = self.model.predict(state)

        return act_values

    def fit(self, batch_x, batch_y, epochs=1, verbose=0):

        fit_result = self.model.fit(batch_x, batch_y, epochs=1, verbose=0)

        #if self.fit_count % Config.Optimizer.TENSORBOARD_UPDATE_FREQUENCY == 0:
        #    fit_result = self.model.fit(batch_x, batch_y, epochs=1, verbose=0, callbacks=[self.tbCallback])
        #    self.fit_count = 0
        #else:
        #    fit_result = self.model.fit(batch_x, batch_y, epochs=1, verbose=0)

        loss = fit_result.history["loss"][0]
        accuracy = fit_result.history["acc"][0]
        self.fit_count += 1
        return loss, accuracy

    def _compile_model(self):
        raise NotImplementedError

#######################
#
#  CNN
#
class DQN_CNN_Model(BaseModel):
    model_name = 'CNN'

    def __init__(self, action_size, board_size, my_button, opponent_button, model_save_path='.'):
        super(DQN_CNN_Model, self).__init__(DQN_CNN_Model.model_name, my_button, opponent_button, model_save_path)

        # two feature plans. one for board of my button, one for board of opponent's button
        self.input_shape = (2, board_size[0], board_size[1])
        self.input_shape_batch = (1, 2, board_size[0], board_size[1])

        self.learning_rate = Config.Optimizer.LEARNING_RATE
        self.learning_rho = Config.Optimizer.LEARNING_RHO

        self.learning_epsilon = Config.Optimizer.LEARNING_EPSILON

        self.action_size = action_size

        self.tbCallback = TensorBoard(log_dir='./Graph', histogram_freq=0,  
            write_graph=False, write_images=False, write_grads=True)

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        #model.add(Conv2D(128, (2, 2), activation='relu' , input_shape=self.input_shape, data_format="channels_first"))
        model.add(Conv2D(128, (2, 2), activation='relu' , input_shape=self.input_shape, data_format="channels_first"))
        #model.add(BatchNormalization())
        #model.add(Activation('relu'))

        #model.add(Conv2D(128, (2, 2),  activation='relu', data_format="channels_first"))
        model.add(Conv2D(128, (2, 2),  activation='relu', data_format="channels_first"))
        #model.add(LeakyReLU())
        #model.add(BatchNormalization())
        #model.add(Activation('relu'))

        model.add(Conv2D(192, (2, 2),  activation='relu', data_format="channels_first"))
        #model.add(LeakyReLU())
        #model.add(BatchNormalization())
        #model.add(Activation('relu'))

        #model.add(Conv2D(192, (2, 2),  data_format="channels_last"))
        #model.add(BatchNormalization())
        #model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(256))
        #model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Dense(self.action_size, activation='linear'))

        self.model = model
        self._compile_model()
        return model

    def state_conversion(self, state):
        my_button = ((state == self.my_button) * 1)
        opponent_button = ((state == self.opponent_button) * 1)

        feature = np.stack([my_button, opponent_button])
        feature = feature.reshape(self.input_shape_batch)

        return feature

    def _compile_model(self):
        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate), metrics=["accuracy"])

        logger.info('compile model')
        self.model.summary(print_fn=logger.info)
