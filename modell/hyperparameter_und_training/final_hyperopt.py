from inspect import ClosureVars
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
    # =====================================================================
    print(f"\n- Create Logger ({datetime.now()})\n")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    fhandler = logging.FileHandler(filename= 'hyperopt_log.txt', mode='a')
    logger.addHandler(fhandler)

    # =====================================================================
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

    # =====================================================================
    # Set parameters and stuff
    print(f"\n- Create values for hyperparameter tuning ({datetime.now()})\n")
    space = {'learning_rate': hp.loguniform("learning_rate", np.log(0.01), np.log(1)),
         'max_delta_step': hp.uniform("max_delta_step", 0.0, 5.0),
         'max_depth': hp.quniform("max_depth", 3, 20, 1),
         'min_child_weight': hp.uniform("min_child_weight", 0.0, 5.0),
         'num_rounds': hp.quniform("num_rounds", 10, 500, 1),
         'reg_alpha': hp.uniform("reg_alpha", 0.0, 5.0),
         'reg_lambda': hp.uniform("reg_lambda", 0.0, 5.0),
         'weight': hp.uniform("weight", 0.0, 0.75),
         }
    space['tree_method'] = "gpu_hist"
    space['objective'] = 'binary:logistic'
    space['eval_metric'] = 'logloss'
    space['sampling_method'] = 'gradient_based'

    # =====================================================================
    print(f"\n- Load objective function ({datetime.now()})\n")
    def mcc_metric(predt: np.ndarray, 
                   dtrain: xgb.DMatrix) -> Tuple[str, float]:
        y = dtrain.get_label()
        mcc = matthews_corrcoef(y, np.round(predt))
        return "MCC_Metric", float(-np.mean(mcc))

    def objective(space):
        space = copy.copy(space)
        num_rounds = space["num_rounds"]
        weight = space["weight"]
        weights = np.array([space["weight"] if binding == 0 else 1.0 for binding in df_train["outcome"]])
        del space["num_rounds"]
        del space["weight"]
        space["max_depth"] = int(space["max_depth"])
        dtrain = xgb.DMatrix(np.array(train_X), 
                            weight = weights, 
                            label = np.array(train_y),
                            feature_names= feature_names)
        cv = xgb.cv(params=space, 
                    dtrain=dtrain, 
                    nfold=5, 
                    num_boost_round=int(num_rounds), 
                    feval=mcc_metric,
                    as_pandas=True)
        # print("Accuracy:", cv.iloc[-1,2])
        print(cv.iloc[-1])
        return {'loss': cv.iloc[-1,6], 'status': STATUS_OK}

    # =====================================================================
    print(f"\n- Start hyperparameter tuning ({datetime.now()})\n")
    trials = Trials()
    n = 20000
    for i in range(10, n+1, 10):
        best_hyperparams = fmin(fn = objective,
                                space = space,
                                algo = tpe.suggest,
                                max_evals = i,
                                trials = trials)
        print(f"\nFound hyperparameter:\n{best_hyperparams}\n")
        np.save("results.npy", trials.best_trial)
        np.save("results.npy", trials.argmin)
        logging.info(i)
        logging.info(trials.best_trial["result"]["loss"])
        logging.info(trials.argmin)

if __name__ == "__main__":
    main()