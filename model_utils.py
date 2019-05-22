import os
import random as rn

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Input, Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def set_seed_global(random_seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(random_seed)
    rn.seed(random_seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    tf.set_random_seed(random_seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def train_keras_model(features, labels, layers, model_name, scheduler=(), lr=0.001,
                      epochs=2, batch_size=128, random_seed=None):
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        random_state=random_seed)
    input_dense = Input((features.shape[1],))
    output = input_dense
    for l in layers:
        output = l(output)
    model = Model(inputs=[input_dense], outputs=output)
    model.compile(optimizer=Adam(lr=lr), loss=binary_crossentropy)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=scheduler)

    pred = model.predict(X_test)
    print_results(y_test, pred, model_name, True)
    return model


def train_xgb(features, labels, model_name, random_seed=None):
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        random_state=random_seed)

    model = XGBClassifier(objective="multi:softprob", num_class=2, seed=random_seed)
    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)[:, 1]
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
        name += " test"
    else:
        name += " whole"
    print(name + " accuracy", accuracy_score(true, pred > 0.5))
    print(name + " ROC AUC", roc_auc_score(true, pred))
