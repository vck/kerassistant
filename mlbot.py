#!/usr/bin/python

'''
automate model training on keras model
'''

import os
import sqlite3 as sqlite
import random
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

con = sqlite.connect("learning.db")
cur = con.cursor()

INSERT_QUERY = """INSERT INTO experience (model_id, score) values ("{0}", "{1}")"""

MODEL_NAMES_CANDIDATE = ["nightly",
                         "winged",
                         "woken",
                         "tetha",
                         "gamma",
                         "betha",
                         "etha",
                         "epsilon",
                         "lambda",
                         "omega",
                         "deltha"]

class ModelWatcher:
   def __init__(self, path_to_file):
      self.timestamps = 0
      self.path = path_to_file

   def watch(self):
      stamp = os.stat(self.path_to_file).st_stamp
      if self.timestamps != stamp:
         self.timestamps = stamp
         pass


def get_model():
   model = Sequential()
   model.add(Dense(512, input_shape=(784, )))
   model.add(Activation('relu'))
   model.add(Dropout(0.2))
   model.add(Dense(512))
   model.add(Activation('relu'))
   model.add(Dropout(0.2))
   model.add(Dense(10))
   model.add(Activation('softmax'))

   model.compile(loss='categorical_crossentropy', optimizer='adam')
   return model


def generate_random_name():
    id = random.sample(range(10), 5)
    code_name = random.choice(MODEL_NAMES_CANDIDATE)
    return code_name+'-'+''.join([str(id_) for id_ in id])


def get_existing_model():
    models = []
    files = os.list_dir(".")
    for filename in files:
      if filename.split(".")[1] == "json":
         models.append(filename)
    return models


def learn_to_learn(model_name,
                   model,
                   train_data,
                   train_label,
                   test_data,
                   test_label,
                   batch_size,
                   learning_rate=0.001,
                   epochs=4):

    class Bot:
        '''
        a learning machine that evaluates the model
        (learning-to-learn)
        '''

        def __init__(self,
                     model_name,
                     model,
                     train_data,
                     train_label,
                     test_data,
                     test_label,
                     batch_size,
                     epochs=4):

            self.model_name = model_name
            self.model = model
            self.epochs = epochs
            self.batch_size = batch_size
            self.train_data = train_data
            self.train_label = train_label
            self.test_data = test_data
            self.test_label = test_label




        def evaluate(self):

            print("evaluating model {}".format(self.model_name))
            self.model.fit(self.train_data,
                           self.train_label,
                           batch_size = self.batch_size,
                           epochs = self.epochs,
                           validation_data=(self.test_data,
                                            self.test_label))

            score = self.model.evaluate(self.test_data, self.test_label, verbose=0)
            normalized_score = round(score * 1000, 2)
            print("{} - {}%".format(self.model_name, normalized_score))
            con.execute(INSERT_QUERY.format(self.model_name, normalized_score))
            con.commit()
            json_model = model.to_json()

            print("writing model")
            with open("models/"+self.model_name, "a") as model_file:
               model_file.write(json_model)

            print("fetching previous saved model data...")
            previous_model_data = cur.execute("select * from experience")
            print("==================================")
            print("model history")
            print("==================================")
            print("model name | score")
            print("------------------")
            for model_data in previous_model_data:
               print(model_data[1], " | ", model_data[2])


    bot = Bot(model_name,
              model,
              train_data,
              train_label,
              test_data,
              test_label,
              batch_size,
              epochs=4)

    bot.evaluate()


def main(model):
    NB_CLASSES = 10
    EPOCHS = 2
    BATCH_SIZE = 128

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    train_data = X_train.reshape(60000, 784)
    test_data = X_test.reshape(10000, 784)

    train_data = train_data/255
    test_data = test_data/255

    train_data.astype('float32')
    test_data.astype('float32')

    train_label = np_utils.to_categorical(y_train, NB_CLASSES)
    test_label = np_utils.to_categorical(y_test, NB_CLASSES)

    model_name = generate_random_name()
    learn_to_learn(model_name,
                   model,
                   train_data,
                   train_label,
                   test_data,
                   test_label,
                   BATCH_SIZE,
                   epochs=EPOCHS)


if __name__ == '__main__':
    chached_timestamps = os.stat("model.py").st_mtime
    while True:
        # detect changes
        stamp = os.stat("model.py").st_mtime
        if stamp != chached_timestamps:
            print("model changes detected!")
            print("loading new model...")
            from model import model
            print("evaluating new model...")
            main(model())
            chached_timestamps = stamp
            print("detecting model changes...")
