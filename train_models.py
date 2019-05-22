import os
from copy import deepcopy

import numpy as np
import tensorflow as tf
import random as rn
from keras.layers import Dense, BatchNormalization
from keras.activations import relu, sigmoid
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K

from preprocess import preprocess
from model_utils import train_keras_model, train_xgb, save_keras_model, save_xgb_model

random_seed = 12345
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(random_seed)
rn.seed(random_seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(random_seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


if __name__ == "__main__":

    file_name = '/home/aga/Fizyka/licencjat/htt_features_test.pkl'

    print("Reading data")
    only_disc_dict = preprocess(file_name, False, True, random_seed=random_seed)
    without_disc_dict = preprocess(file_name, True, False, random_seed=random_seed)
    whole_data_dict = preprocess(file_name, True, True, random_seed=random_seed)
    without_BCI_dict = preprocess(
        file_name, True, True, random_seed=random_seed,
        mask_columns=['leg_2_byCombinedIsolationDeltaBetaCorrRaw3Hits'],
    )
    datas = [whole_data_dict, without_disc_dict, without_BCI_dict]

    # only disc
    layers = [Dense(1, activation=sigmoid)]
    name = "only_disc"
    print("#" * 100)
    print("Training " + name)
    model = train_keras_model(only_disc_dict["features"], only_disc_dict["labels"],
                              layers, name, epochs=4, random_seed=random_seed)
    save_keras_model(model, only_disc_dict, name)

    # old model
    layers = [
        Dense(32, activation=relu),
        Dense(32, activation=relu),
        Dense(32, activation=relu),
        Dense(32, activation=relu),
        Dense(32, activation=relu),
        Dense(32, activation=relu),
        Dense(1, activation=sigmoid)
    ]
    names = ["whole_data", "without_disc", "without_BCI"]
    for name, data_dict in zip(names, datas):
        print("#" * 100)
        print("Training " + name)
        model = train_keras_model(data_dict["features"], data_dict["labels"],
                                  deepcopy(layers), name, epochs=4,
                                  random_seed=random_seed)
        save_keras_model(model, data_dict, name)

    # new model
    layers = [
        Dense(256, activation=sigmoid),
        BatchNormalization(),
        Dense(256, activation=sigmoid),
        Dense(1, activation=sigmoid),
    ]
    scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1,
                                  min_lr=0.00001)
    names = ["new_whole_data", "new_without_disc", "new_without_BCI"]
    for name, data_dict in zip(names, datas):
        print("#" * 100)
        print("Training " + name)
        model = train_keras_model(data_dict["features"], data_dict["labels"],
                                  deepcopy(layers), name, scheduler=[scheduler],
                                  lr=0.0005, epochs=6, batch_size=256,
                                  random_seed=random_seed)
        save_keras_model(model, data_dict, name)

    # xgb model
    names = ["xgb_whole_data", "xgb_without_disc", "xgb_without_BCI"]
    for name, data_dict in zip(names, datas):
        print("#" * 100)
        print("Training " + name)
        model = train_xgb(data_dict["features"], data_dict["labels"], name, random_seed)
        save_xgb_model(model, data_dict, name)

