import collections
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from keras import backend as K
import keras.models as km
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
import os.path
from ..utils.logger import logger
from ..config.config import Config


class BaseModel():
    def __init__(self, model_name, model_save_path):
        self.model_save_path = model_save_path
        self.model_name = model_name

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

        loss = fit_result.history["loss"][0]
        accuracy = fit_result.history["acc"][0]
        return loss, accuracy

    def _compile_model(self):
        raise NotImplementedError

####################
#
#  simple NN model
#
class DQNModel(BaseModel):
    model_name = 'NN_128x16'

    def __init__(self, action_size, board_size, model_save_path='.'):
        super(DQNModel, self).__init__(DQNModel.model_name, model_save_path)

        self.learning_rate = Config.Optimizer.LEARNING_RATE
        self.input_dim = np.prod(board_size)

        self.action_size = action_size


    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        self.model = model
        self._compile_model()
        return model

    def _compile_model(self):
        self.model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate),
            metrics=["accuracy"])

        logger.info('compile model')
        self.model.summary(print_fn=logger.info)

    def state_conversion(self, state):
        state = state.reshape([1, self.input_dim])

        return state


#######################
#
#  CNN
#
class DQN_CNN_Model(BaseModel):
    model_name = 'CNN_38x74x158'

    def __init__(self, action_size, board_size, model_save_path='.'):
        super(DQN_CNN_Model, self).__init__(DQN_CNN_Model.model_name, 
                                            model_save_path)

        self.input_shape = (board_size[0], board_size[1], 1)
        self.input_shape_batch = (1, board_size[0], board_size[1], 1)

        self.learning_rate = Config.Optimizer.LEARNING_RATE
        self.learning_rho = Config.Optimizer.LEARNING_RHO

        self.learning_epsilon = Config.Optimizer.LEARNING_EPSILON

        self.action_size = action_size

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(
            Conv2D(
                38, (4, 4),
                activation="relu",
                input_shape=self.input_shape,
                data_format="channels_last"))
        model.add(
            Conv2D(74, (3, 3), activation="relu", data_format="channels_last"))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(158, activation="relu"))
        model.add(Dropout(0.45))
        model.add(Dense(self.action_size, activation='linear'))

        self.model = model
        self._compile_model()
        return model

    def state_conversion(self, state):
        state = state.reshape(self.input_shape_batch)
        return state

    def _compile_model(self):
        self.model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate),
            metrics=["accuracy"])

        logger.info('compile model')
        self.model.summary(print_fn=logger.info)


# Deep Q-learning Agent
class DQNAgent():
    '''
    The agent assumes he is holding '-1'  button,  the env is holding '+1'  when training

    if now when the agent is asked to hold '+1' button, we flip all the state variables from -1 to 1 and 1 to -1
    '''

    def __init__(self,
                 who,
                 model_name,
                 load_model=False,
                 save_learnt_to_file=False,
                 board_size=(6, 7),
                 action_size=7):

        self.board_size = board_size
        self.action_size = action_size
        self.save_learnt_to_file = save_learnt_to_file

        self.memory_normal  = collections.deque(maxlen=Config.Optimizer.NORMAL_MEMORY_SIZE)
        self.memory_winning = collections.deque(maxlen=Config.Optimizer.WINNING_MEMORY_SIZE)
        self.memory_lossing = collections.deque(maxlen=Config.Optimizer.LOSSING_MEMORY_SIZE)
        self.memory_important = collections.deque(maxlen=Config.Optimizer.IMPORTANT_MEMORY_SIZE)

        self.gamma = Config.Explorer.GAMMA  # discount rate
        self.epsilon = Config.Explorer.EPSILON  # exploration rate
        self.epsilon_mid = Config.Explorer.EPSILON_MID
        self.epsilon_decay_to_mid = Config.Explorer.EPSILON_DECAY_TO_MID
        self.epsilon_min = Config.Explorer.EPSILON_MIN
        self.epsilon_decay_to_min = Config.Explorer.EPSILON_DECAY_TO_MIN

        self.fitting_cb = None

        ## sample 33% from important queue
        self.start_training_size = Config.Optimizer.START_TRAINING_SIZE
        self.batch_normal = Config.Optimizer.BATCH_NORMAL
        self.batch_winning = Config.Optimizer.BATCH_WINNING
        self.batch_lossing = Config.Optimizer.BATCH_LOSSING
        self.batch_important = Config.Optimizer.BATCH_IMPORTANT
        self.batch_size = Config.Optimizer.BATCH_SIZE

        if who == 'player':
            self.button_color_invert = 1  # to multiple the state by this varible. meaning no change
        else:
            # who == 'npc'
            self.button_color_invert = -1  # to multiple the state by this varible. meaning -1 to 1 , 1 to -1

        model_save_path = Config.Folder.TRAINED_FOLDER
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        models = {
            DQNModel.model_name: DQNModel,
            DQN_CNN_Model.model_name: DQN_CNN_Model
        }

        if model_name is None:
            model_name = DQN_CNN_Model.model_name

        self.model_creator = lambda : models[model_name](action_size , self.board_size, model_save_path=model_save_path)
        #self.model = models[model_name](action_size , self.board_size, model_save_path=model_save_path)
        self.model = self.model_creator()

        if load_model:
            logger.info('agent with button ' + str(who) + ' is loading model')
            model_exist = self.model.load_model_from_file()

            if not model_exist:
                logger.info('New model file will be created while learn')
                self.model.build_model()
        else:
            self.model.build_model()


    def add_fitting_callback(self, cb):
        self.fitting_cb = cb

    def _remember(self, state, action, reward, next_state, done):
        self.memory_normal.append((state, action, reward, next_state, done))

        if  reward < 0.0:  # lossing
            self.memory_lossing.append((state, action, reward, next_state, done))
        elif reward > 0.0 :  # winning or no space
            self.memory_winning.append((state, action, reward, next_state, done))
        else:
            num_played_button = np.count_nonzero( state )
            if (not done) and (num_played_button >= Config.Optimizer.NUM_BUTTON_PLAYED_AS_IMPORTANT ):
                self.memory_important.append((state, action, reward, next_state, done))



    def _flip_state(self, state):
        board = state

        board = board.copy()

        if self.button_color_invert == -1:
            board = board * self.button_color_invert

        return board

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            return action

        #state = ( self.board.copy() , act_row, act_col )
        state = self._flip_state(state)
        state = self.model.state_conversion(state)
        act_values = self.model.predict(state)

        action = np.argmax(act_values[0])  # returns action

        return action

    def learn(self, state, action, reward, next_state, done):
        #logger.info('state')
        #logger.info(state)
        #logger.info('next state')
        #logger.info(next_state)
        state = self._flip_state(state)
        state = self.model.state_conversion(state)
        next_state = self._flip_state(next_state)
        next_state = self.model.state_conversion(next_state)

        self._remember(state, action, reward, next_state, done)
        self._replay(done)

    def _get_target_state_action_value(self, next_state):
        predict_next_state_action_values = self.model.predict(next_state)
        return predict_next_state_action_values

    def _get_training_x_y(self, state, action, reward, next_state, done):
        target = reward

        if not done:
            predict_next_state_action_values = self._get_target_state_action_value(
                next_state)

            max_next_state_action_value = np.amax(
                predict_next_state_action_values[0])

            target = reward + self.gamma * max_next_state_action_value

        target_f = self.model.predict(state)

        target_f[0][action] = target

        max_q = np.max(target_f[0])

        return state, target_f, max_q


    def _check_enough_memory(self):
        memory_normal_size = len(self.memory_normal)
        memory_winning_size = len(self.memory_winning)
        memory_lossing_size = len(self.memory_lossing)
        memory_important_size = len(self.memory_important)

        is_enough = ((memory_normal_size>= self.start_training_size) and 
                    (memory_winning_size>= self.start_training_size) and 
                    (memory_lossing_size>= self.batch_lossing) and (memory_important_size>= self.batch_important) )

        return is_enough

    def _sample_memory(self):
        minibatch_1 = random.sample(self.memory_normal, self.batch_normal)
        minibatch_2 = random.sample(self.memory_winning, self.batch_winning)
        minibatch_3 = random.sample(self.memory_lossing, self.batch_lossing)
        minibatch_4 = random.sample(self.memory_important, self.batch_important)

        minibatch = minibatch_1 + minibatch_2 + minibatch_3 + minibatch_4
        random.shuffle(minibatch)

        return minibatch

    def _replay(self, done):

        memory_enough = self._check_enough_memory()
        # if done or (memory_size >= batch_size ):
        if done and memory_enough :
            #minibatch = random.sample(self.memory, batch_size)

            minibatch = self._sample_memory()

            train_x = []
            train_y = []
            max_q = []
            for state, action, reward, next_state, done in minibatch:
                x, y, q = self._get_training_x_y(state, action, reward,
                                                 next_state, done)

                train_x.append(x)
                train_y.append(y)
                max_q.append(q)

            batch_x = np.concatenate(train_x)
            batch_y = np.concatenate(train_y)

            #self.model.fit(x, y, epochs=1, verbose=0)
            loss, accuracy = self.model.fit(
                batch_x, batch_y, epochs=1, verbose=0)
            mean_q = np.mean(max_q)

            if self.fitting_cb is not None:
                self.fitting_cb(loss, accuracy, mean_q)

            if self.epsilon > self.epsilon_mid:
                self.epsilon *= self.epsilon_decay_to_mid
            elif self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_to_min

    def save_model(self):
        if self.save_learnt_to_file:
            self.model.save_model()
        else:
            logger.info('This agent is not for saving new learning to file')


class DDQNAgent(DQNAgent):
    # Double Deep Q-learning Agent
    '''
    The agent assumes he is holding '-1'  button,  the env is holding '+1'  when training

    if now when the agent is asked to hold '+1' button, we flip all the state variables from -1 to 1 and 1 to -1
    '''

    def __init__(self,
                 who,
                 model_name,
                 load_model=False,
                 save_learnt_to_file=False,
                 board_size=(6, 7),
                 action_size=7):
        super(DDQNAgent, self).__init__(
            who=who,
            model_name=model_name,
            load_model=load_model,
            save_learnt_to_file=save_learnt_to_file,
            board_size=board_size,
            action_size=action_size)


        ### create target network model
        self.target_model = self.model_creator()

        logger.info('build target model')
        self.target_model.build_model()

        self.update_target_network()

    def _get_target_state_action_value(self, next_state):
        predict_next_state_action_values = self.target_model.predict(
            next_state)
        return predict_next_state_action_values

    def update_target_network(self):
        self.target_model.model.set_weights(self.model.model.get_weights())
