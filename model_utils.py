import os
import random as rn

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Input, Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from logger import logger


class LossHistory(keras.callbacks.Callback):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        idx = np.random.randint(0, len(self.y), 128)
        loss = self.model.evaluate(self.X[idx], self.y[idx], verbose=0)
        self.val_losses.append(loss)


def set_seed_global(random_seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(random_seed)
    rn.seed(random_seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    tf.set_random_seed(random_seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def train_keras_model(features, labels, layers, model_name, scheduler=None, lr=0.001,
                      epochs=2, batch_size=128, random_seed=None):
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        random_state=random_seed)
    history = LossHistory(X_test, y_test)
    if scheduler is None:
        scheduler = [history]
    else:
        scheduler.append(history)
    input_dense = Input((features.shape[1],))
    output = input_dense
    for l in layers:
        output = l(output)
    model = Model(inputs=[input_dense], outputs=output)
    model.compile(optimizer=Adam(lr=lr), loss=binary_crossentropy)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              callbacks=scheduler, validation_data=(X_test, y_test))

    np.save('models/train_losses_' + model_name, np.array(history.losses))
    np.save('models/val_losses_' + model_name, np.array(history.val_losses))
    pred = model.predict(X_test)
    print_results(y_test, pred, model_name, True)
    return model


def split(features, labels, random_seed):
    return train_test_split(features, labels, random_state=random_seed)


def save_xgb_losses(model, X_train, X_test, y_train, y_test, model_name):
    losses_train = []
    losses_test = []
    for i in range(1, 101):
        pred_train = model.predict_proba(X_train, ntree_limit=i)
        pred_test = model.predict_proba(X_test, ntree_limit=i)
        losses_train.append(log_loss(y_train, pred_train))
        losses_test.append(log_loss(y_test, pred_test))
    np.save("models/train_losses_" + model_name, losses_train)
    np.save("models/val_losses_" + model_name, losses_test)


def train_xgb(features, labels, model_name, random_seed=None):
    X_train, X_test, y_train, y_test = split(features, labels, random_seed)

    model = XGBClassifier(objective="multi:softprob", num_class=2, seed=random_seed)
    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)[:, 1]
    save_xgb_losses(model, X_train, X_test, y_train, y_test, model_name)
    print_results(y_test, pred, model_name, True)
    return model


def save_keras_model(model, data_dict, model_name):
    pred_whole = model.predict(data_dict["not_shuffled_features"])
    print_results(data_dict["not_shuffled_labels"], pred_whole, model_name, False)
    np.save('models/pred_' + model_name, pred_whole)
    model.save_weights("models/" + model_name + ".h5")
    model.save("models/cpp_" + model_name + ".h5", include_optimizer=False)


def save_xgb_model(model, data_dict, model_name):
    pred_whole = model.predict_proba(data_dict["not_shuffled_features"])[:, 1]
    print_results(data_dict["not_shuffled_labels"], pred_whole, model_name, False)
    bst = model.get_booster()
    bst.dump_model("models/cpp_" + model_name)
    model.save_model("models/" + model_name)
    np.save('models/pred_' + model_name, pred_whole)


def print_results(true, pred, model_name, is_test):
    name = model_name
    if is_test:
        name += ", test"
    else:
        name += ", whole"
    res_acc = name + " accuracy " + str(accuracy_score(true, pred > 0.5))
    res_auc = name + " ROC AUC " + str(roc_auc_score(true, pred))
    with open("results.txt", "a") as f:
        f.write(name + ',' + str(roc_auc_score(true, pred)))
        f.write("\n")
    logger.info(res_acc)
    logger.info(res_auc)
