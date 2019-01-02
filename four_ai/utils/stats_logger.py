import os
import csv
import numpy as np
import shutil
import matplotlib
import matplotlib.pyplot as plt
import datetime
from .logger import logger

matplotlib.use("Agg")

#TRAINING_UPDATE_FREQUENCY = 1000
#RUN_UPDATE_FREQUENCY = 10
TRAINING_UPDATE_FREQUENCY = 5000
RUN_UPDATE_FREQUENCY =  100
MAX_LOSS = 5


class StatsLogger:

    def __init__(self, header, directory_path):
        directory_path = directory_path
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path, ignore_errors=True)
        os.makedirs(directory_path)

        self.score = Stat("run", "score", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.step = Stat("run", "step", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.loss = Stat("update", "loss", TRAINING_UPDATE_FREQUENCY, directory_path, header)
        self.accuracy = Stat("update", "accuracy", TRAINING_UPDATE_FREQUENCY, directory_path, header)
        self.q = Stat("update", "q", TRAINING_UPDATE_FREQUENCY, directory_path, header)

    def add_run(self, run):

        #if run % RUN_UPDATE_FREQUENCY == 0:
        #    logger.info('{{"metric": "run", "value": {}}}'.format(run))
        return 

    def add_score(self, run, score):
        self.score.add_entry(run, score)

    def add_step(self, run, step):
        self.step.add_entry(run, step)

    def add_accuracy(self, fit, accuracy):
        self.accuracy.add_entry(fit, accuracy)

    def add_loss(self, fit, loss):
        loss = min(MAX_LOSS, loss)  # Loss clipping for very big values that are likely to happen in the early stages of learning
        self.loss.add_entry(fit, loss)

    def add_q(self, fit, q):
        self.q.add_entry(fit, q)

    def log_iteration(self, episode, reward , t):
        self.add_run(episode)
        self.add_score(episode, reward)
        self.add_step(episode, t)

    def log_fitting(self, fit_time, loss,accuracy, q):
        self.add_accuracy(fit_time, accuracy)
        self.add_loss(fit_time, loss)
        self.add_q(fit_time,q)

    def save_csv(self):
        self.score.save_file()
        self.step.save_file()
        self.loss.save_file()
        self.accuracy.save_file()
        self.q.save_file()


class Stat:

    def __init__(self, x_label, y_label, update_frequency, directory_path, header):
        self.x_label = x_label
        self.y_label = y_label
        self.update_frequency = update_frequency
        self.directory_path = directory_path
        self.header = header
        self.temp_y_values = []
        self.xy_values = []

        self.csv_path = self.directory_path + self.y_label + '.csv'
        self.png_path = self.directory_path + self.y_label + '.png'

    def save_file(self):
        self._append_save_csv(self.csv_path, self.xy_values)
        self.xy_values = []


    def save_png(self):
        self._save_png(input_path=self.csv_path,
                        output_path=self.png_path,
                        small_batch_length=self.update_frequency,
                        big_batch_length=self.update_frequency*10,
                        x_label=self.x_label,
                        y_label=self.y_label)

    def add_entry(self, x_value, y_value):
        self.temp_y_values.append(y_value)
        if len(self.temp_y_values) % self.update_frequency == 0:
            mean_value = np.mean(self.temp_y_values)
            min_value = min(self.temp_y_values)
            max_value = max(self.temp_y_values)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            self.xy_values.append( [ x_value, mean_value , min_value, max_value , timestamp ] )

            self.temp_y_values = []

    def _save_png(self, input_path, output_path, small_batch_length, big_batch_length, x_label, y_label):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)

            for i in range(0, len(data)):
                #x.append(float(i)*small_batch_length)
                x.append(float(data[i][0]))
                y.append(float(data[i][1]))

        plt.subplots()
        plt.plot(x, y, label="last " + str(small_batch_length) + " average")

        '''
        batch_averages_y = []
        batch_averages_x = []
        temp_values_in_batch = []
        relative_batch_length = big_batch_length/small_batch_length

        for i in range(len(y)):
            temp_values_in_batch.append(y[i])
            if (i+1) % relative_batch_length == 0:
                if not batch_averages_y:
                    batch_averages_y.append(mean(temp_values_in_batch))
                    batch_averages_x.append(0)
                batch_averages_x.append(len(batch_averages_y)*big_batch_length)
                batch_averages_y.append(mean(temp_values_in_batch))
                temp_values_in_batch = []
        if len(batch_averages_x) > 1:
            plt.plot(batch_averages_x, batch_averages_y, linestyle="--", label="last " + str(big_batch_length) + " average")
        '''

        # if len(x) > 1:
        #     trend_x = x[1:]
        #     z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
        #     p = np.poly1d(z)
        #     plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _append_save_csv(self, path, xy_values):
        if not os.path.exists(path):
            with open(path, "w", newline=''):
                pass

        scores_file = open(path, "a" , newline='')
        with scores_file:
            writer = csv.writer(scores_file)

            writer.writerows(xy_values)