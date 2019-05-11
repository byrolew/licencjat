from copy import copy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt


def get_data(filename, pred_dict):
    models = ['leg_2_deepTau2017v1tauVSjet']
    legs, jets, global_params, properties = pd.read_pickle(filename)
    y_true = np.array(global_params["sampleType"])
    y_true = np.reshape(y_true, (-1, 1))
    y_preds = {model: copy(properties[model]) for model in models}
    for name, file in pred_dict.items():
        y_preds[name] = np.load(file)
    return y_true, y_preds


def save_plot(y_true, y_preds, name, xlim=(0.2, 0.8), ylim=(1e-5, 1e-2)):
    plt.figure(figsize=(10, 7))
    for model, y_pred in y_preds.items():
        print('ROC AUC score for {} model: '.format(model), roc_auc_score(y_true, y_pred))
        roc = roc_curve(y_true, y_pred)
        plt.semilogy(roc[1], roc[0], linetype[model.strip(" bez_disc")],
                     label=model.strip(" bez_disc"), linewidth=2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.legend()
    plt.savefig(name, dpi=500)


if __name__ == "__main__":
    linetype = {
        'leg_2_deepTau2017v1tauVSjet': 'C0:',
        'Same klasyfikatory': 'C1--',
        'XGBoost': 'C2-',
        'Nowa sieć neuronowa': 'C3-',
        'Stara sieć neuronowa': 'C4-.',
        'Nowa sieć neuronowa bez dyskryminatorów': 'C1--',
        'Nowa sieć neuronowa bez głównej zmiennej': 'C4-.',
    }

    pred_dict = {
        "Same klasyfikatory": "models/pred_only_disc.npy",
        "Stara sieć neuronowa": "models/pred_whole_data.npy",
        "Stara sieć neuronowa bez_disc": "models/pred_without_disc.npy",
        "XGBoost": "models/pred_xgb_whole_data.npy",
        "XGBoost bez_disc": "models/pred_xgb_without_disc.npy",
        "Nowa sieć neuronowa": "models/pred_new_whole_data.npy",
        "Nowa sieć neuronowa bez dyskryminatorów": "models/pred_new_without_disc.npy",
        "Nowa sieć neuronowa bez_disc": "models/pred_new_without_disc.npy",
        "Nowa sieć neuronowa bez głównej zmiennej": "models/pred_new_without_BCI.npy",
    }
    plot1 = {"Same klasyfikatory", "Stara sieć neuronowa",
             "XGBoost", "Nowa sieć neuronowa", "leg_2_deepTau2017v1tauVSjet"}
    plot2 = {"Stara sieć neuronowa bez_disc", "leg_2_deepTau2017v1tauVSjet",
             "XGBoost bez_disc", "Nowa sieć neuronowa bez_disc"}
    plot3 = {"leg_2_deepTau2017v1tauVSjet", "Nowa sieć neuronowa",
             "Nowa sieć neuronowa bez dyskryminatorów",
             "Nowa sieć neuronowa bez głównej zmiennej"}

    y_true, y_preds = get_data("htt_features_test.pkl", pred_dict)

    save_plot(y_true, {k: v for k, v in y_preds.items() if k in plot1},
              name="best_models.png", ylim=(5e-6, 1e-2))
    save_plot(y_true, {k: v for k, v in y_preds.items() if k in plot2},
              name="without_disc.png", ylim=(1e-4, 1e-2))
    save_plot(y_true, {k: v for k, v in y_preds.items() if k in plot3},
              name="new_network.png")
