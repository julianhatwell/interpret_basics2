import json
import time
import timeit
import numpy as np # check if need *. so far it's only np.unique in f_survey routine
from pandas import DataFrame
from forest_surveyor.plotting import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def tune_rf(X, y, grid = None, random_state=123, save_path = None):

    # to do - test allowable structure of grid input

    if grid is None:
        grid = ParameterGrid({
            'n_estimators': [(i + 1) * 500 for i in range(3)]
            , 'max_depth' : [i for i in [8, 16]]
            , 'min_samples_leaf' : [1, 5]
            })

    start_time = timeit.default_timer()

    rf = RandomForestClassifier()
    params = []
    best_score = 0

    print(str(len(grid)) + ' runs to do')
    print()
    for g in grid:
        print('starting new run at: ' + time.asctime(time.gmtime()))
        print(g)
        tree_start_time = timeit.default_timer()
        rf.set_params(oob_score = True, random_state=random_state, **g)
        rf.fit(X, y)
        tree_end_time = timeit.default_timer()
        g['elapsed_time'] = tree_end_time - tree_start_time
        g['score'] = rf.oob_score_
        print('ending run at: ' + time.asctime(time.gmtime()))
        params.append(g)
        print('completed ' + str(len(params)) + ' run(s)')
        print()
        if rf.oob_score_ > best_score:
            best_score = rf.oob_score_
            best_grid = g

    elapsed = timeit.default_timer() - start_time
    print('Time elapsed:', "{:0.4f}".format(elapsed), 'seconds')

    if save_path is not None:
        with open(save_path + 'params.json', 'w') as outfile:
            json.dump(params, outfile)

    return(params)

def train_rf(X, y, params = None, encoder = None, random_state = 123):

    # to do - test allowable structure of grid input
    if params is not None:
        params = DataFrame(params)

        # get the best params
        best_grid = params.loc[params['score'].idxmax()]
        best_params = {k: int(v) for k, v in best_grid.items() if k not in ('score', 'elapsed_time')}
        print("Best OOB Cohen's Kappa during tuning: " "{:0.4f}".format(best_grid.score))
        print("Best parameters:", best_params)
        print()
        print("Training a random forest model using best parameters... (please wait)")
        rf = RandomForestClassifier(random_state=random_state, oob_score=True, **best_params)
    else:
        print("Training a random forest model using default parameters... (please wait)")
        rf = RandomForestClassifier(random_state=random_state, oob_score=True)

    rf.fit(X, y)
    print()
    print("Done")
    print()

    if encoder is not None:
        enc_model = make_pipeline(encoder, rf)
        print("Created helper function enc_model(). A pipeline: feature encoding -> rf model")
        return(rf, enc_model)
    else:
        print('No encoder was provided. An encoder is required if not all the data is numerical.')
        return(rf)

def evaluate_model(prediction_model, X, y, class_names=None):
    pred = prediction_model.predict(X)
    print("Cohen's Kappa on unseen instances: " "{:0.4f}".format(cohen_kappa_score(y, pred)))

    # view the confusion matrix
    cm = confusion_matrix(y, pred)
    plot_confusion_matrix(cm, class_names=class_names,
                          title='Confusion matrix, without normalization')
    # normalized confusion matrix
    plot_confusion_matrix(cm
                          , class_names=class_names
                          , normalize=True,
                          title='Normalized confusion matrix')

def forest_survey(f_walker, X, y):

    if f_walker.encoder is not None:
        X = f_walker.encoder.transform(X)

    f_walker.full_survey(X, y)
    return(f_walker.forest_stats(np.unique(y)))

def cor_incor_split(prediction_model, X, y):
    correct_preds = prediction_model.predict(X) == y
    incorrect_preds = prediction_model.predict(X) != y

    if sum(correct_preds) > 0:
        X_cor = X[correct_preds.values]
        y_cor = y[correct_preds.values]
    else:
        X_cor = None
        y_cor = None

    if sum(incorrect_preds) > 0:
        X_incor = X[incorrect_preds.values]
        y_incor = y[incorrect_preds.values]
    else:
        X_incor = None
        y_incor = None
    return(X_cor, y_cor, X_incor, y_incor)

def cor_incor_forest_survey(f_walker, X, y):
    X_c, y_c, X_i, y_i = cor_incor_split(f_walker.prediction_model, X, y)

    if X_c is not None:
        f_cor_stats = forest_survey(f_walker, X_c, y_c)
    else:
        f_cor_stats = None

    if X_i is not None:
        f_incor_stats = forest_survey(f_walker, X_i, y_i)
    else:
        f_incor_stats = None

    return(f_cor_stats, f_incor_stats)

# def walk_paths(f_walker, X, y = None, by_tree=True):
