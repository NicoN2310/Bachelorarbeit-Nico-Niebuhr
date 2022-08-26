import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, matthews_corrcoef

best_hyperparams = {'learning_rate': 0.22205346352424176, 'max_delta_step': 2.1132866218907003, 'max_depth': 9, 'min_child_weight': 1.9215635292274276, 'num_rounds': 779.304784864205, 'reg_alpha': 0.6126926526952088, 'reg_lambda': 3.238607289889072, 'weight': 0.24922052982680407}


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
                         feature_names= feature_names
                         )

    dtest = xgb.DMatrix(np.array(test_X), 
                        label = np.array(test_y),
                        feature_names= feature_names
                        )

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
    pass