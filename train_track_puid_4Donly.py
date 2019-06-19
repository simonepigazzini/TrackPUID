#!/usr/bin/env python

import os
import xgboost
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

import argparse

features = [
    'pt',
    'eta',
    'phi',
    'chi2',
    'ndof',
    'numberOfValidHits',
    'numberOfValidPixelBarrelHits',
    'numberOfValidPixelEndcapHits',
    'dt',
    'sigmat0',
    'btlMatchChi2',
    'btlMatchTimeChi2',
    'etlMatchChi2',
    'etlMatchTimeChi2',
    'mtdt',
    'path_len'
]

## command_line options
parser = argparse.ArgumentParser(description='Run adversarial training of CMS Hgg diphoton ID MVA')
parser.add_argument('--inp-dir', type=str, dest='inp_dir', default='', help='input directory')
parser.add_argument('--out-dir', type=str, dest='out_dir', default='/tmp/spigazzi/MTD/TDR/TrackPUID/results/')
parser.add_argument('--inp-file', type=str, dest='inp_file', default='input_tracks_train.hd5')
parser.add_argument('--n-thread', type=int, dest='n_thread', default=8)
parser.add_argument('--wmtd', action='store_true', dest='wmtd', default=False)
parser.add_argument('--bo', action='store_true', dest='do_bo', default=False)
    
## parse options
options = parser.parse_args()
print(options)

df = pd.read_hdf(options.inp_dir+'/'+options.inp_file)

if options.wmtd:
    df['dt'] = df['t0']-df['pv_t']
    
target = 'simIsFromPV'
selection = (df['dz']<0.1) & (df['sigmat0']>0)
df = df[selection]

if options.do_bo:
    dtrain = xgboost.DMatrix(df[features], label=df[target], weight=df['weight'])
    del(df)

    early_stops = []
    def train_clf(min_child_weight, colsample_bytree, max_depth, subsample, gamma, reg_alpha, reg_lambda):
        res = xgboost.cv(
            {
                'min_child_weight': min_child_weight,
                'colsample_bytree': colsample_bytree, 
                'max_depth': int(max_depth),
                'subsample': subsample,
                'gamma': gamma,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'objective' : 'binary:logistic', 
                'tree_method' : 'hist',
                'eval_metric': 'auc'            
            },
            dtrain, num_boost_round=2000, nfold=5, early_stopping_rounds=100
        )
        early_stops.append(len(res))

        value = res['test-auc-mean'].iloc[-1]
    
        return value

    hyperpar =  {
        'min_child_weight': (5, 25),
        'colsample_bytree': (0.5, 1),
        'max_depth': (15, 30),
        'subsample': (0.8, 1),
        'gamma': (5, 15),
        'reg_alpha': (15, 30),
        'reg_lambda': (15, 30)
    }

    xgb_bo = BayesianOptimization(train_clf, hyperpar)    

    xgb_bo.maximize(init_points=3, n_iter=20, acq='ei')

    params = xgb_bo.max['params']

    # generate train and validation indices
    indices = np.arange(dtrain.num_row())
    np.random.shuffle(indices)
    valid_idx = indices[:int(dtrain.num_row()/10)]
    train_idx = indices[int(dtrain.num_row()/10)+1:]
    validation_res = {}
    clf = xgboost.train(
        {
            'min_child_weight' : params['min_child_weight'],
            'colsample_bytree' : params['colsample_bytree'],
            'max_depth' : int(params['max_depth']),
            'subsample' : params['subsample'],
            'gamma' : params['gamma'],
            'reg_alpha' : params['reg_alpha'],
            'reg_lambda' : params['reg_lambda'],
            'objective' : 'binary:logistic', 
            'tree_method' : 'hist',
            'eval_metric' : 'auc'
        },
        dtrain.slice(train_idx), early_stops[np.argmax(list(map(lambda v : v["target"], xgb_bo.res)))],
        evals=[(dtrain.slice(valid_idx), 'valid')], evals_result=validation_res, early_stopping_rounds=50
    )
    
    clfname = 'clf4D' if options.wmtd else 'clf3D'
    outfile = open(options.out_dir+'/track_puid_'+clfname+'.pkl', 'wb')
    pickle.dump(clf, outfile)
    pickle.dump(validation_res, outfile)
    pickle.dump(xgb_bo.res, outfile)

else:
    clfname = 'clf4D' if options.wmtd else 'clf3D'
    with open(options.out_dir+'/track_puid_'+clfname+'.pkl', 'rb') as bofile:
        booster = pickle.load(bofile)
        results = pickle.load(bofile)
        bo_res = pickle.load(bofile)

    params = bo_res[np.argmax(list(map(lambda v : v["target"], bo_res)))]['params']
    config = {
        'min_child_weight' : params['min_child_weight'],
        'colsample_bytree' : params['colsample_bytree'],
        'max_depth' : int(params['max_depth']),
        'subsample' : params['subsample'],
        'gamma' : params['gamma'],
        'reg_alpha' : params['reg_alpha'],
        'reg_lambda' : params['reg_lambda'],
        'objective' : 'binary:logistic', 
        'tree_method' : 'hist'
    }

    X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(df[features], df[target], df['weight'], test_size=0.2, random_state=12345)
    clf = xgboost.XGBClassifier(n_estimators=1000, **config)
    bst = clf.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_valid, y_valid)], sample_weight_eval_set=[w_valid], eval_metric='auc',
                  early_stopping_rounds=100)
    valid_result = clf.evals_result()
    best_n_trees = bst.best_ntree_limit
    
    outfile = open(options.out_dir+'/track_puid_optimized_'+clfname+'.pkl', 'wb')
    pickle.dump(clf, outfile)
    pickle.dump(best_n_trees, outfile)
    pickle.dump(valid_result, outfile)
    

