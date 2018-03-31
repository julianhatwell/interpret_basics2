import json
import time
import timeit
import pickle
import numpy as np
import multiprocessing as mp
from pandas import DataFrame
from forest_surveyor.plotting import plot_confusion_matrix
from forest_surveyor.structures import rule_accumulator
from forest_surveyor.mp_callable import mp_run_rf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def tune_rf_mp(X, y, grid = None, random_state=123, save_path = None):

    if grid is None:
        grid = ParameterGrid({
            'n_estimators': [(i + 1) * 500 for i in range(3)]
            , 'max_depth' : [i for i in [8, 16]]
            , 'min_samples_leaf' : [1, 5]
            })

    start_time = timeit.default_timer()

    # Define an output queue
    output = mp.Queue()

    # List of processes that we want to run
    processes = [mp.Process(target=mp_run_rf, args=(X, y, g, random_state, output)) for g in grid]

    print(str(len(grid)) + ' runs to do')
    print()
    print('Going parallel...')

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    params = [output.get() for p in processes]

    print('Completed ' + str(len(params)) + ' run(s) and exited parallel')
    print()
    elapsed = timeit.default_timer() - start_time
    print('Time elapsed:', "{:0.4f}".format(elapsed), 'seconds')

    if save_path is not None:
        with open(save_path + 'params.json', 'w') as outfile:
            json.dump(params, outfile)

    params = DataFrame(params).sort_values(['score','n_estimators','max_depth','min_samples_leaf'],
                                            ascending=[False, True, True, False])
    return(params)

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

    params = DataFrame(params).sort_values(['score','n_estimators','max_depth','min_samples_leaf'],
                                            ascending=[False, True, True, False])
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

def run_batches(f_walker, getter,
 data_container, sample_instances, sample_labels,
 batch_size = 1, n_batches = 1, alpha= 0.5):
    results = [[]] * (batch_size * n_batches)
    best_rule = [[]] * (batch_size * n_batches)
    for b in range(n_batches):
        instances, labels = getter.get_next(batch_size)

        walked = f_walker.forest_walk(instances = instances
                                , labels = labels)

        # rearrange paths by instances
        walked.flip()

        for i in range(batch_size):
            walked.set_paths(i-1, which_trees='majority')
            walked.discretize_paths(data_container.var_dict)
            walked.set_patterns()
            ra = rule_accumulator(data_container=data_container, paths_container=walked)
            ra.profile(sample_instances=sample_instances, sample_labels=sample_labels)
            ra.prune_rule()

            results[b * batch_size + i] = [
            [p[ra.target_class] for p in ra.pri_and_post], # should be the same as precision below
            [c[ra.target_class] for c in ra.pri_and_post_coverage],
            [a[ra.target_class] for a in ra.pri_and_post_accuracy],
            ra.precision,
            ra.isolation_pos,
            len(ra.pruned_rule)]

            _, score1, score2 = ra.score_rule(alpha)
            adj_max_score1 = np.max(score1[ra.isolation_pos:])
            score1_loc = np.where(np.array(score1 == adj_max_score1))[0][0]
            adj_max_score2 = np.max(score2[ra.isolation_pos:])
            score2_loc = np.where(np.array(score2 == adj_max_score2))[0][0]

            ra_best = rule_accumulator(data_container=data_container, paths_container=walked)
            ra_best.profile(sample_instances=sample_instances, sample_labels=sample_labels, fixed_length=score2_loc)
            ra_best.prune_rule()

            best_rule[b * batch_size + i] = [ra_best.pruned_rule, adj_max_score1, adj_max_score2, len(ra_best.pruned_rule)]

    results_store = open(data_container.pickle_path('results.pickle'), "wb")
    pickle.dump(results, results_store)
    results_store.close()

    best_rule_store = open(data_container.pickle_path('best_rule.pickle'), "wb")
    pickle.dump(best_rule, best_rule_store)
    best_rule_store.close()

    return(ra, results, best_rule)
