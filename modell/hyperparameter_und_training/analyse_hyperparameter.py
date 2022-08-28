import xgboost as xgb
import pandas as pd
import numpy as np

from matplotlib import pyplot

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import matthews_corrcoef

hyperparams = {}


def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b"][ind]
        ecfp = np.array(list(df["ECFP"][ind])).astype(int)
                
        X = X +(np.concatenate([ecfp, emb]), )
        y = y + (df["outcome"][ind], )

    return(np.array(X), np.array(y))

def create_model(df_train):
    train_X, train_y =  create_input_and_output_data(df = df_train)
    
    feature_names =  ["ECFP_" + str(i) for i in range(1024)]
    feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]

    hyperparams['objective'] = 'binary:logistic'
    hyperparams['eval_metric'] = 'error'
    hyperparams['tree_method'] = "gpu_hist"
    hyperparams['sampling_method'] = 'gradient_based'

    num_rounds = hyperparams["num_rounds"]
    weights = np.array([hyperparams["weight"] if binding == 0 else 1.0 for binding in df_train["outcome"]])
    hyperparams["max_depth"] = int(hyperparams["max_depth"])
    del hyperparams["num_rounds"]
    del hyperparams["weight"]
    
    dtrain = xgb.DMatrix(
        np.array(train_X), 
        weight = weights, 
        label = np.array(train_y),
        feature_names= feature_names
    )

    bst = xgb.train(
        hyperparams, 
        dtrain, 
        int(num_rounds)
    )

    return bst

def main():
    df_train = pd.read_pickle("training_data.pkl")
    df_test = pd.read_pickle("test_data.pkl")