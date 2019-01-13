
class BaseModel():
    def __init__(self, model_name, my_button, opponent_button, model_save_path):
        self.model_save_path = model_save_path
        self.model_name = model_name

        self.my_button = my_button
        self.opponent_button = opponent_button
        self.fit_count = 0

    def save_model_backup_copy(self, backup_name):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model_from_file(self):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def fit(self, batch_x, batch_y, epochs=1, verbose=0):
        raise NotImplementedError

    def _compile_model(self):
        raise NotImplementedError