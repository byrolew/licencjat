from copy import copy
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt


def get_data(filename, pred_dict):
    model = 'leg_2_deepTau2017v1tauVSjet'
    legs, jets, global_params, properties = pd.read_pickle(filename)
    y_true = np.array(global_params["sampleType"])
    y_true = np.reshape(y_true, (-1, 1))
    y_preds = {"deepTau": copy(properties[model])}
    for name, file in pred_dict.items():
        y_preds[name] = np.load(file)
    return y_true, y_preds


def save_plot(y_true, y_preds, auc_dict, name, xlim=(0.2, 0.8), ylim=(1e-5, 1e-2)):
    plt.figure(figsize=(8, 5))
    for model, y_pred in y_preds.items():
        print('ROC AUC score for {} model: '.format(model), roc_auc_score(y_true, y_pred))
        roc = roc_curve(y_true, y_pred)
        plt.semilogy(
            roc[1], roc[0], linetype[re.sub(" bez_disc$", "", model)],
            label=re.sub(" bez_disc$", "", model) + " " + auc_dict[model],
            linewidth=2
        )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.legend()
    plt.savefig("praca/" + name, dpi=400)


if __name__ == "__main__":
    linetype = {
        'deepTau': 'C0:',
        'ensemble': 'C1--',
        'XGBoost': 'C2-',
        'best-nn': 'C3-',
        'baseline': 'C4-.',
        'best-nn bez klasyfikatorów': 'C1--',
        'best-nn bez głównej zmiennej': 'C4-.',
    }

    pred_dict = {
        "ensemble": "models/pred_only_disc.npy",
        "baseline": "models/pred_whole_data.npy",
        "baseline bez_disc": "models/pred_without_disc.npy",
        "XGBoost": "models/pred_xgb_whole_data.npy",
        "XGBoost bez_disc": "models/pred_xgb_without_disc.npy",
        "best-nn": "models/pred_new_whole_data.npy",
        "best-nn bez klasyfikatorów": "models/pred_new_without_disc.npy",
        "best-nn bez_disc": "models/pred_new_without_disc.npy",
        "best-nn bez głównej zmiennej": "models/pred_new_without_BCI.npy",
    }

    auc_dict = {
        "deepTau": "[0.9945]",
        "ensemble": "[0.9956]",
        "baseline": "[0.9948]",
        "baseline bez_disc": "[0.9940]",
        "XGBoost": "[0.9985]",
        "XGBoost bez_disc": "[0.9956]",
        "best-nn": "[0.9979]",
        "best-nn bez klasyfikatorów": "[0.9949]",
        "best-nn bez_disc": "[0.9949]",
        "best-nn bez głównej zmiennej": "[0.9972]",
    }

    plot1 = {"ensemble", "baseline",
             "XGBoost", "best-nn", "deepTau"}
    plot2 = {"baseline bez_disc", "deepTau",
             "XGBoost bez_disc", "best-nn bez_disc"}
    plot3 = {"deepTau", "best-nn",
             "best-nn bez klasyfikatorów",
             "best-nn bez głównej zmiennej"}

    y_true, y_preds = get_data("htt_features_test.pkl", pred_dict)

    save_plot(y_true, {k: v for k, v in y_preds.items() if k in plot1}, auc_dict,
              name="best_models.png", ylim=(5e-6, 1e-2))
    save_plot(y_true, {k: v for k, v in y_preds.items() if k in plot2}, auc_dict,
              name="without_disc.png", ylim=(1e-4, 1e-2))
    save_plot(y_true, {k: v for k, v in y_preds.items() if k in plot3}, auc_dict,
              name="new_network.png")
