import json
import time
import timeit
import pickle
import numpy as np
import multiprocessing as mp
from pandas import DataFrame
from forest_surveyor.plotting import plot_confusion_matrix
from forest_surveyor.structures import rule_accumulator, forest_walker, batch_getter
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
    try:
        with open(save_path + 'params.json', 'r') as infile:
            params = json.load(infile)
        print('Using existing params file. To re-tune, delete file at ' + save_path + 'params.json')
        print()
        return(params)
    except:
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
        print()
        return(rf, enc_model)
    else:
        print('No encoder was provided. An encoder is required if not all the data is numerical.')
        print()
        return(rf, rf)

def evaluate_model(prediction_model, X, y, class_names=None, plot_cm=True, plot_cm_norm=True):
    pred = prediction_model.predict(X)
    print("Cohen's Kappa on unseen instances: " "{:0.4f}".format(cohen_kappa_score(y, pred)))

    # view the confusion matrix
    cm = confusion_matrix(y, pred)
    if plot_cm:
        plot_confusion_matrix(cm, class_names=class_names,
                              title='Confusion matrix, without normalization')
    # normalized confusion matrix
    if plot_cm_norm:
        plot_confusion_matrix(cm
                              , class_names=class_names
                              , normalize=True,
                              title='Normalized confusion matrix')
    return(cm)

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
 batch_size = 1, n_batches = 1,
 support_paths=0.1, alpha_paths=0.5,
 alpha_scores=0.5, which_trees='majority'):
    sample_rule_accs = [[]] * n_batches
    results = [[]] * (batch_size * n_batches)
    for b in range(n_batches):

        instances, labels = getter.get_next(batch_size)
        instance_ids = labels.index.tolist()
        # get all the tree paths instance by instance
        walked = f_walker.forest_walk(instances = instances
                                , labels = labels)

        # rearrange paths by instances
        walked.flip()

        for i in range(batch_size):

            # process the path info for freq patt mining
            walked.set_paths(i-1, which_trees=which_trees)
            walked.discretize_paths(data_container.var_dict)
            walked.set_patterns(support=support_paths, alpha=alpha_paths)

            # grow a maximal rule from the freq patts
            ra = rule_accumulator(data_container=data_container, paths_container=walked, instance_id=instance_ids[i])
            ra.profile(sample_instances=sample_instances, sample_labels=sample_labels)
            ra.prune_rule()

            # score the rule at each additional term
            _, score1, score2 = ra.score_rule(alpha=alpha_scores) # not the same as the paths alpha, only affects score one
            adj_max_score1 = np.max(score1[ra.isolation_pos:])
            score1_loc = np.where(np.array(score1 == adj_max_score1))[0][0]
            adj_max_score2 = np.max(score2[ra.isolation_pos:])
            score2_loc = np.where(np.array(score2 == adj_max_score2))[0][0]

            # re-run the profile to the best scoring fixed length
            ra_best = rule_accumulator(data_container=data_container, paths_container=walked, instance_id=instance_ids[i])
            ra_best.profile(sample_instances=sample_instances, sample_labels=sample_labels, fixed_length=score2_loc)
            ra_best.prune_rule()

            # collect results
            results[b * batch_size + i] = [ra_best.lite_instance(), ra.lite_instance()]

        # saving a full rule_accumulator object at the end of each batch, for plotting etc
        sample_rule_accs[b] = ra
        # report progress
        print('done batch ' + str(b))

    results_store = open(data_container.pickle_path('results.pickle'), "wb")
    pickle.dump(results, results_store)
    results_store.close()

    return(sample_rule_accs, results)

def experiment(get_dataset, n_instances, n_batches,
 support_paths=0.1,
 alpha_paths=0.5,
 which_trees='majority'):

    print('LOADING NEW DATA SET.')
    print()
    # load a data set
    mydata = get_dataset()

    # train test split
    tt = mydata.tt_split()

    ################ PARAMETER TUNING ###################
    ############ Only runs when required ################
    #####################################################

    print('Finding best paramaters for Random Forest. Checking for prior tuning paramters.')
    print()
    params = tune_rf(tt['X_train_enc'], tt['y_train'],
     save_path = mydata.pickle_path(),
     random_state=mydata.random_state)

    #####################################################

    # train a rf model
    rf, enc_rf = train_rf(X=tt['X_train_enc'], y=tt['y_train'],
     params=params,
     encoder=tt['encoder'],
     random_state=mydata.random_state)

    # fit the forest_walker
    f_walker = forest_walker(forest = rf,
     features=mydata.onehot_features,
     encoder=tt['encoder'],
     prediction_model=enc_rf)

    # run the batch based forest walker
    getter = batch_getter(instances=tt['X_test'], labels=tt['y_test'])

    # faster to do one batch, avoids the overhead of setting up many but consumes more mem
    batch_size = int(min(n_instances, len(tt['y_test'])) / n_batches)

    print('Starting new run at: ' + time.asctime(time.gmtime()) + ' with batch_size = ' + str(batch_size) + ' and n_batches = ' + str(n_batches) + '...(please wait)')
    start_time = timeit.default_timer()

    # rule_acc is just the last rule rule_accumulator, results are for the whole batch
    rule_acc, results, best_rule = run_batches(f_walker=f_walker,
     getter=getter,
     data_container=mydata,
     sample_instances=tt['X_train_enc'],
     sample_labels=tt['y_train'],
     support_paths=support_paths,
     alpha_paths=alpha_paths,
     which_trees=which_trees,
     batch_size = batch_size,
     n_batches = n_batches)

    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    print('Done. Completed run at: ' + time.asctime(time.gmtime()) + '. Elapsed time (seconds) = ' + str(elapsed_time))
    print()

    print('Results saved at ' + mydata.pickle_path('results.pickle'))
    print()
    print('To retrieve results execute the following:')
    print('results_store = open(\'' + mydata.pickle_path('results.pickle') + '\', "rb")')
    print('results = pickle.load(results_store)')
    print('results_store.close()')
    print()
    print()
    return(rule_acc, results)
