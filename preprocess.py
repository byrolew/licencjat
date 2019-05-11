from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd


def preprocess(file_name, enable_data=True, enable_dicriminators=True,
               mask_columns=(), random_seed=None):
    np.random.seed(random_seed)
    legs, jets, global_params, properties = pd.read_pickle(file_name)
    properties = OrderedDict(sorted(properties.items(), key=lambda t: t[0]))

    sampleType = np.array(global_params["sampleType"])
    sampleType = np.reshape(sampleType, (-1, 1))
    features = np.array(list(properties.values()))
    features = np.transpose(features)
    feature_names = list(properties.keys())

    # Redefine DPF output to be 1 for signal
    discName = "leg_2_DPFTau_2016_v1tauVSall"
    DPF_index = feature_names.index(discName)
    features[:, DPF_index] *= -1
    features[:, DPF_index] += 1
    indexes = features[:, DPF_index] > 1
    features[indexes, DPF_index] = 0.0
    # Filter features to be usedfor training
    column_mask = np.full(features.shape[1], enable_data)
    oldMVA_discriminators = [
        "leg_2_byIsolationMVArun2v1DBnewDMwLTraw2017v2",
        "leg_2_DPFTau_2016_v1tauVSall",
        "leg_2_deepTau2017v1tauVSall",
        "leg_2_deepTau2017v1tauVSjet",
    ]
    for discName in oldMVA_discriminators:
        index = feature_names.index(discName)
        column_mask[index] = enable_dicriminators

    for col_name in mask_columns:
        index = feature_names.index(col_name)
        column_mask[index] = False

    features = features[:, column_mask]
    not_shuffled_features = deepcopy(features)
    not_shuffled_labels = deepcopy(sampleType)

    features = np.hstack((sampleType, features))
    np.random.shuffle(features)

    labels = features[:, 0]
    features = features[:, 1:]

    # print("Input data shape:", features.shape)
    # print("Number of positive examples:", (labels > 0.5).sum())
    # print("Number of negative examples:", (labels < 0.5).sum())

    assert features.shape[0] == labels.shape[0]

    tmp = np.array(feature_names)
    tmp = tmp[column_mask]
    feature_names = list(tmp)
    return {
        "features": features,
        "not_shuffled_features": not_shuffled_features,
        "labels": labels,
        "not_shuffled_labels": not_shuffled_labels,
        "feature_names": feature_names,
    }
