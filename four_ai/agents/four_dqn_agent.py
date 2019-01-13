import collections
import numpy as np
import random
import os.path
from ..utils.logger import logger
from ..config.config import Config
from .model.keras_model import DQN_CNN_Model


# Deep Q-learning Agent
class DQNAgent():
    '''
    The agent assumes he is holding '-1'  button,  the env is holding '+1'  when training

    if now when the agent is asked to hold '+1' button, we flip all the state variables from -1 to 1 and 1 to -1
    '''

    def __init__(self, who, model_name, load_model=False, save_learnt_to_file=False, board_size=(6, 7), action_size=7):

        self.board_size = board_size
        self.action_size = action_size
        self.save_learnt_to_file = save_learnt_to_file

        self.memory_normal = collections.deque(maxlen=Config.Optimizer.NORMAL_MEMORY_SIZE)
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
        self.my_button = -1
        self.opponent_button = 1

        if who == 'player':
            self.button_color_invert = 1  # to multiple the state by this varible. meaning no change
        else:
            # who == 'npc'
            self.button_color_invert = -1  # to multiple the state by this varible. meaning -1 to 1 , 1 to -1

        self.who_am_i = who

        model_save_path = Config.Folder.TRAINED_FOLDER
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        models = {DQN_CNN_Model.model_name: DQN_CNN_Model}

        if model_name is None:
            model_name = DQN_CNN_Model.model_name

        self.model_creator = lambda: models[model_name](action_size, self.board_size, self.my_button , self.opponent_button, model_save_path=model_save_path)
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

    def _remember(self, state, action, reward, next_state, done, scn):
        self.memory_normal.append((state, action, reward, next_state, done))

        if Config.Optimizer.MULTI_MEMORY_QUEUE:
            # special queue for wrong move
            if done and (scn == 'player_wrong_move'):
                self.memory_important.append((state, action, reward, next_state, done))

            if reward < 0.0:  # lossing
                self.memory_lossing.append((state, action, reward, next_state, done))
            elif reward > 0.0:  # winning or no space
                self.memory_winning.append((state, action, reward, next_state, done))

    def _flip_state(self, state):
        board = state

        board = board.copy()

        if self.button_color_invert == -1:
            board = board * self.button_color_invert

        return board

    def _find_easy_loss(self, others_button, b_width, b_height, board, avail_row, in_row_count):

        for others_action in range(b_width):
            ## pretend player acton
            done, act_row, act_col, new_board, new_avail_row = self._test_move(others_action, others_button, avail_row,
                                                                               board, b_width, b_height, in_row_count)

            if act_row == -1:  # wrong move
                continue

            # check win condition
            if done:
                ## player will win, npc should move here to defend
                return True, others_action

        return False, None

    def _find_easy_win(self, my_button, b_width, b_height, board, avail_row, in_row_count):

        for my_action in range(b_width):
            # trial move npc button
            done, act_row, act_col, new_board, new_avail_row = self._test_move(my_action, my_button, avail_row, board,
                                                                               b_width, b_height, in_row_count)

            if act_row == -1:  # wrong move
                continue

            # check win condition
            if done:
                ## win
                return True, my_action

        return False, None

    def _get_avail_row(self, board, b_width, b_height):
        def find_avail(column, b_height):
            for index in range(b_height - 1, -1, -1):
                element = column[index]
                if element == 0:
                    return index

            return -1

        avail_row = np.zeros(b_width).astype(int)
        for i in range(0, b_width):
            avail_row[i] = find_avail(board[:, i], b_height)

        return avail_row

    def _default_action(self, state):
        b_height = self.board_size[0]
        b_width = self.board_size[1]
        in_row_count = 4
        cur_board = state
        avail_row = self._get_avail_row(cur_board, b_width, b_height)

        found, action = self._find_easy_win(self.my_button, b_width, b_height, cur_board, avail_row, in_row_count)
        if found:
            return found, action

        found, action = self._find_easy_loss(self.opponent_button, b_width, b_height, cur_board, avail_row,
                                             in_row_count)
        if found:
            return found, action

        return False, None

    def _test_move(self, act_col, act_button, cur_avail_row, cur_board, b_width, b_height, in_row_count):
        avail_row = cur_avail_row.copy()
        board = cur_board.copy()

        done = False
        # the column is full
        if avail_row[act_col] == -1:
            act_row = -1  # wrong move
            return done, act_row, act_col, board, avail_row

        board[avail_row[act_col], act_col] = act_button
        act_row = avail_row[act_col]
        avail_row[act_col] -= 1

        done = self._check_win(act_button, act_row, act_col, board, b_width, b_height, in_row_count)
        return done, act_row, act_col, board, avail_row

    def _check_win(self, act_button, act_row, act_col, board, b_width, b_height, in_row_count):
        # check horizonal
        for i in range(0, b_width - in_row_count + 1):
            if np.sum(board[act_row, i:i + in_row_count]) == (act_button * in_row_count):
                return True

        # check vertical
        for i in range(0, b_height - in_row_count + 1):
            if np.sum(board[i:i + in_row_count, act_col]) == (act_button * in_row_count):
                return True

        # check diagonal
        # from bottom left to top right
        for i in range(in_row_count - 1, -1, -1):
            row0 = act_row + i
            col0 = act_col - i

            row1 = row0 - in_row_count + 1
            col1 = col0 + in_row_count - 1

            if (row1 >= 0) and (row0 < b_height) and (col0 >= 0) and (col1 < b_width):
                total_b = 0
                for j in range(in_row_count):
                    irow = row0 - j
                    icol = col0 + j

                    total_b += board[irow, icol]

                if total_b == (act_button * in_row_count):
                    return True

        # check diagonal
        # from top left to bottom right
        for i in range(in_row_count - 1, -1, -1):
            row0 = act_row - i
            col0 = act_col - i

            row1 = row0 + in_row_count - 1
            col1 = col0 + in_row_count - 1

            if (row0 >= 0) and (row1 < b_height) and (col0 >= 0) and (col1 < b_width):
                total_b = 0
                for j in range(in_row_count):
                    irow = row0 + j
                    icol = col0 + j

                    total_b += board[irow, icol]

                if total_b == (act_button * in_row_count):
                    return True

        return False

    def act(self, state):

        state = self._flip_state(state)

        ### look for obvious win or loss situtation
        has_default_action, default_action = self._default_action(state)

        if has_default_action:
            return default_action

        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            return action

        state = self.model.state_conversion(state)
        act_values = self.model.predict(state)

        action = np.argmax(act_values[0])  # returns action

        return action

    def learn(self, state, action, reward, next_state, done, scn):
        #logger.info('state')
        #logger.info(state)
        #logger.info('next state')
        #logger.info(next_state)
        state = self._flip_state(state)
        state = self.model.state_conversion(state)
        next_state = self._flip_state(next_state)
        next_state = self.model.state_conversion(next_state)

        self._remember(state, action, reward, next_state, done, scn)
        self._replay(done)

    def _get_target_state_action_value(self, next_state):
        predict_next_state_action_values = self.model.predict(next_state)
        return predict_next_state_action_values

    def _get_training_x_y(self, state, action, reward, next_state, done):
        target = reward

        if not done:
            predict_next_state_action_values = self._get_target_state_action_value(next_state)

            max_next_state_action_value = np.amax(predict_next_state_action_values[0])

            target = reward + self.gamma * max_next_state_action_value

        target_f = self.model.predict(state)

        target_f[0][action] = target

        max_q = np.max(target_f[0])

        return state, target_f, max_q

    def _check_enough_memory(self):

        if Config.Optimizer.MULTI_MEMORY_QUEUE:
            memory_normal_size = len(self.memory_normal)
            memory_winning_size = len(self.memory_winning)
            memory_lossing_size = len(self.memory_lossing)
            memory_important_size = len(self.memory_important)

            is_enough = ((memory_normal_size >= self.start_training_size)
                         and (memory_winning_size >= self.start_training_size)
                         and (memory_lossing_size >= self.batch_lossing)
                         and (memory_important_size >= self.batch_important))
        else:
            memory_normal_size = len(self.memory_normal)
            is_enough = (memory_normal_size >= self.start_training_size)

        return is_enough

    def _sample_memory(self):
        if Config.Optimizer.MULTI_MEMORY_QUEUE:
            minibatch_1 = random.sample(self.memory_normal, self.batch_normal)
            minibatch_2 = random.sample(self.memory_winning, self.batch_winning)
            minibatch_3 = random.sample(self.memory_lossing, self.batch_lossing)
            minibatch_4 = random.sample(self.memory_important, self.batch_important)

            minibatch = minibatch_1 + minibatch_2 + minibatch_3 + minibatch_4
            random.shuffle(minibatch)
        else:
            minibatch = random.sample(self.memory_normal, self.batch_size)

        return minibatch

    def _replay(self, done):

        memory_enough = self._check_enough_memory()
        # if done or (memory_size >= batch_size ):
        if done and memory_enough:
            #minibatch = random.sample(self.memory, batch_size)

            minibatch = self._sample_memory()

            train_x = []
            train_y = []
            max_q = []
            for state, action, reward, next_state, done in minibatch:
                x, y, q = self._get_training_x_y(state, action, reward, next_state, done)

                train_x.append(x)
                train_y.append(y)
                max_q.append(q)

            batch_x = np.concatenate(train_x)
            batch_y = np.concatenate(train_y)

            #self.model.fit(x, y, epochs=1, verbose=0)
            loss, accuracy = self.model.fit(batch_x, batch_y, epochs=1, verbose=0)
            mean_q = np.mean(max_q)

            if self.fitting_cb is not None:
                self.fitting_cb(loss, accuracy, mean_q)

            if self.epsilon > self.epsilon_mid:
                self.epsilon *= self.epsilon_decay_to_mid
            elif self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_to_min

    def save_model(self, backup_copy_name=None):
        if backup_copy_name is None:
            if self.save_learnt_to_file:
                self.model.save_model()
            else:
                logger.info('This agent is not for saving new learning to file')
        else:
            self.model.save_model_backup_copy(backup_copy_name)


class DDQNAgent(DQNAgent):
    # Double Deep Q-learning Agent
    '''
    The agent assumes he is holding '-1'  button,  the env is holding '+1'  when training

    if now when the agent is asked to hold '+1' button, we flip all the state variables from -1 to 1 and 1 to -1
    '''

    def __init__(self, who, model_name, load_model=False, save_learnt_to_file=False, board_size=(6, 7), action_size=7):
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
        predict_next_state_action_values = self.target_model.predict(next_state)
        return predict_next_state_action_values

    def update_target_network(self):
        self.target_model.model.set_weights(self.model.model.get_weights())
