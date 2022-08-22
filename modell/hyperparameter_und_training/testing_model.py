import xgboost as xgb
import pandas as pd
import numpy as np
import copy
from datetime import datetime
import logging
from typing import Tuple

from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, roc_auc_score, matthews_corrcoef
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe, rand

def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b"][ind]
        ecfp = np.array(list(df["ECFP"][ind])).astype(int)
                
        X = X +(np.concatenate([ecfp, emb]), )
        y = y + (df["outcome"][ind], )

    return(X,y)

def main():
    print(f"\n- Preparing data for tuning/training/testing ({datetime.now()})\n")
    df_train = pd.read_pickle("training_data.pkl")
    df_test = pd.read_pickle("test_data.pkl")

    train_X, train_y =  create_input_and_output_data(df = df_train)
    test_X, test_y =  create_input_and_output_data(df = df_test)

    train_X = np.array(train_X)
    test_X  = np.array(test_X)

    train_y = np.array(train_y)
    test_y  = np.array(test_y)

    feature_names =  ["ECFP_" + str(i) for i in range(1024)]
    feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]

    # =========================================================================
    print(f"\n- Start training with best hyperparameters ({datetime.now()})\n")
    best_hyperparams = {'learning_rate': 0.89890843032133091, 'max_delta_step': 3.1389633747760453, 'max_depth': 14, 'min_child_weight': 3.459838301755225, 'num_rounds': 232.08438067605385, 'reg_alpha': 3.9276571933444133, 'reg_lambda': 0.13838704730326057, 'weight': 0.006050435961558444}
    # LogLoss - Random
    # {'learning_rate': 0.04260770450325235, 'max_delta_step': 4.50077219973334, 'max_depth': 6, 'min_child_weight': 2.1692664195440545, 'num_rounds': 331.2952013431535, 'reg_alpha': 1.9886864492952605, 'reg_lambda': 0.1656211721547718, 'weight': 0.0006740345130188352}
        
    # MCC - Random
    # {'learning_rate': 0.03214030570602307, 'max_delta_step': 0.36662438223933114, 'max_depth': 10.0, 'min_child_weight': 1.5093557977721401, 'num_rounds': 61.0, 'reg_alpha': 1.3167068759149587, 'reg_lambda': 1.3499763946639214, 'weight': 0.1990442381429972}
    # {'learning_rate': 0.03214030570602307, 'max_delta_step': 0.36662438223933114, 'max_depth': 10.0, 'min_child_weight': 1.5093557977721401, 'num_rounds': 61.0, 'reg_alpha': 1.3167068759149587, 'reg_lambda': 1.3499763946639214, 'weight': 0.1990442381429972}
    
    # LogLoss - TPE
    # {'learning_rate': 0.8540824703052108, 'max_delta_step': 3.8998811315170725, 'max_depth': 9.0, 'min_child_weight': 2.4373594209477174, 'num_rounds': 668.0, 'reg_alpha': 0.6297296757101909, 'reg_lambda': 4.909857682993391, 'weight': 0.16104888651621746}
    # {'learning_rate': 0.9189620405342076, 'max_delta_step': 3.765929527547236, 'max_depth': 15.0, 'min_child_weight': 3.80302135564452, 'num_rounds': 498.0, 'reg_alpha': 0.10850816561483789, 'reg_lambda': 3.9826868884097752, 'weight': 0.19765279921541257}

    print(best_hyperparams)

    best_hyperparams['objective'] = 'binary:logistic'
    # best_hyperparams['eval_metric'] = 'logloss'
    best_hyperparams['eval_metric'] = 'error'
    best_hyperparams['tree_method'] = "gpu_hist"
    best_hyperparams['sampling_method'] = 'gradient_based'

    num_rounds = best_hyperparams["num_rounds"]
    weights = np.array([best_hyperparams["weight"] if binding == 0 else 1.0 for binding in df_train["outcome"]])
    best_hyperparams["max_depth"] = int(best_hyperparams["max_depth"])
    del best_hyperparams["num_rounds"]
    del best_hyperparams["weight"]

    dtrain = xgb.DMatrix(np.array(train_X), 
                         weight = weights, 
                         label = np.array(train_y),
                         feature_names= feature_names)

    dtest = xgb.DMatrix(np.array(test_X), 
                        label = np.array(test_y),
                        feature_names= feature_names)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    print("="*15)
    bst = xgb.train(best_hyperparams, 
                    dtrain, 
                    int(num_rounds), 
                    evallist)
    print("="*15)

    # =====================================================================
    print(f"\n- Start evaluating model ({datetime.now()})\n")
    y_test_pred = np.round(bst.predict(dtest))

    acc_test = np.mean(y_test_pred == np.array(test_y))
    roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
    mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

    print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))
    df_test["prediction"] = y_test_pred
    seq_identity = ["60-80%", "40-60%", "<40%"]

    for identity in seq_identity:
        y_true = np.array(df_test["outcome"].loc[df_test["Sequence identity"] == identity])
        y_pred = np.array(df_test["prediction"].loc[df_test["Sequence identity"] == identity])
        acc = np.mean(y_pred == np.array(y_true))
        mcc = matthews_corrcoef(np.array(y_true), y_pred)
        print("Sequence identity %s, Accuracy: %s, MCC: %s \n" % (identity, acc, mcc))

if __name__ == "__main__":
    main()