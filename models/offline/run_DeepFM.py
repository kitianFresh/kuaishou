
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from  sklearn.metrics import roc_auc_score as roc

import conf.DeepFM_conf as DeepFM_conf
#from metrics import gini_norm
from conf.DeepFM_conf import FeatureDictionary, DataParser

from DeepFM import DeepFM


#gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _load_data():

    dfTrain = pd.read_csv(DeepFM_conf.TRAIN_FILE, sep='\t')
    print(dfTrain.info())
    dfTest = pd.read_csv(DeepFM_conf.TEST_FILE, sep='\t')
    print(dfTest.info())



    cols = [c for c in dfTrain.columns if c not in ["user_id", "photo_id", "click"]]
    cols = [c for c in cols if (not c in DeepFM_conf.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["click"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["user_id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in DeepFM_conf.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    #print(len(folds))
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=DeepFM_conf.NUMERIC_COLS,
                           ignore_cols=DeepFM_conf.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)


    dfTrain, dfTest= data_parser.scale(df_train=dfTrain, df_test=dfTest)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_valid, Xv_valid, y_valid = data_parser.parse(df=dfTest, has_label=True)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    #_get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    roc_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    roc_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)

    roc_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)

    print(len(Xi_train), len(Xi_train[0]), len(Xv_train), len(Xv_train[0]))
    print(len(Xi_valid), len(Xi_valid[0]), len(Xv_valid), len(Xv_valid[0]))
    #print(len(Xi_test), len(Xi_test[0]), len(Xv_test), len(Xi_test[0]))
    print(len(y_train), len(y_valid))
    dfm = DeepFM(**dfm_params)
    dfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
    y_train_meta = dfm.predict(Xi_valid, Xv_valid)

    print("roc: ", roc(dfTest["click"].values, y_test_meta.values))
    #roc_results_cv[i] = roc(y_valid_, y_train_meta[valid_idx])
    #roc_results_epoch_train[i] = dfm.train_result
    #roc_results_epoch_valid[i] = dfm.valid_result



    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, roc_results_cv.mean(), roc_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, roc_results_cv.mean(), roc_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    #_plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"user_id": ids["user_id"],"photo_id": ids["photo_id"], "click": y_pred.flatten()}).to_csv(
        os.path.join(DeepFM_conf.SUB_DIR, filename), index=False, float_format="%.5f")


'''
def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()
'''

# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()
#print(dfTrain.shape)

# folds
folds = list(StratifiedKFold(n_splits=DeepFM_conf.NUM_SPLITS, shuffle=True,
                             random_state=DeepFM_conf.RANDOM_SEED).split(X_train, y_train))

#print(folds)


# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 10,
    "batch_size": 128,
    "learning_rate": 0.0002,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc,
    "random_seed": DeepFM_conf.RANDOM_SEED
}
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

# ------------------ FM Model ------------------
#fm_params = dfm_params.copy()
#fm_params["use_deep"] = False
#y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)


# ------------------ DNN Model ------------------
#dnn_params = dfm_params.copy()
#dnn_params["use_fm"] = False
#y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)



