import json
import time
import timeit
import pickle
import numpy as np
import multiprocessing as mp
from pandas import DataFrame, Series
from forest_surveyor import p_count, p_count_corrected
import forest_surveyor.datasets as ds
from forest_surveyor.plotting import plot_confusion_matrix
from forest_surveyor.structures import rule_accumulator, forest_walker, batch_getter, rule_tester, loo_encoder

from scipy.stats import chi2_contingency
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score

from anchor import anchor_tabular as anchtab
from lime import lime_tabular as limtab

# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


def do_tuning(X, y, grid = None, random_state=123, save_path = None):
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

    for g in grid:
        tree_start_time = timeit.default_timer()
        rf.set_params(oob_score = True, random_state=random_state, **g)
        rf.fit(X, y)
        tree_end_time = timeit.default_timer()
        g['elapsed_time'] = tree_end_time - tree_start_time
        g['score'] = rf.oob_score_
        params.append(g)

    elapsed = timeit.default_timer() - start_time

    params = DataFrame(params).sort_values(['score','n_estimators','max_depth','min_samples_leaf'],
                                        ascending=[False, True, True, False])

    best_grid = params.loc[params['score'].idxmax()]
    best_params = {k: int(v) if k not in ('score', 'elapsed_time') else v for k, v in best_grid.items()}

    if save_path is not None:
        with open(save_path + 'best_params_rndst_' + str(random_state) + '.json', 'w') as outfile:
            json.dump(best_params, outfile)

    return(best_params)

def tune_rf(X, y, grid = None, random_state=123, save_path = None, override_tuning=False):

    # to do - test allowable structure of grid input
    if override_tuning:
        print('overriding previous tuning parameters')
        best_params = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)
    else:
        try:
            with open(save_path + 'best_params_rndst_' + str(random_state) + '.json', 'r') as infile:
                print('using previous tuning parameters')
                best_params = json.load(infile)
            return(best_params)
        except:
            print('finding best tuning parameters')
            best_params = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)

    print("Best OOB Accuracy Estimate during tuning: " "{:0.4f}".format(best_params['score']))
    print("Best parameters:", best_params)
    print()

    return(best_params)

def train_rf(X, y, best_params = None, encoder = None, random_state = 123):

    # to do - test allowable structure of grid input
    if best_params is not None:
        # train a random forest model using given parameters
        best_params = {k: v for k, v in best_params.items() if k not in ('score', 'elapsed_time')}
        rf = RandomForestClassifier(random_state=random_state, oob_score=True, **best_params)
    else:
        # train a random forest model using default parameters
        rf = RandomForestClassifier(random_state=random_state, oob_score=True)

    rf.fit(X, y)

    if encoder is not None:
        # create helper function enc_model(). A pipeline: feature encoding -> rf model
        enc_model = make_pipeline(encoder, rf)
        return(rf, enc_model)
    else:
        # if no encoder provided, return the basic model
        return(rf, rf)

def evaluate_model(prediction_model, X, y, class_names=None, plot_cm=True, plot_cm_norm=True):
    pred = prediction_model.predict(X)

    # view the confusion matrix
    cm = confusion_matrix(y, pred)
    prfs = precision_recall_fscore_support(y, pred)
    acc = accuracy_score(y, pred)
    coka = cohen_kappa_score(y, pred)

    if plot_cm:
        plot_confusion_matrix(cm, class_names=class_names,
                              title='Confusion matrix, without normalization')
    # normalized confusion matrix
    if plot_cm_norm:
        plot_confusion_matrix(cm
                              , class_names=class_names
                              , normalize=True,
                              title='Normalized confusion matrix')
    return(cm, acc, coka, prfs)

def forest_survey(f_walker, X, y):

    if f_walker.encoder is not None:
        X = f_walker.encoder.transform(X)

    f_walker.full_survey(X, y)
    return(f_walker.forest_stats(np.unique(y)))

def run_batches(f_walker, getter,
 data_container, encoder, sample_instances, sample_labels,
 batch_size = 1, n_batches = 1,
 support_paths=0.1, alpha_paths=0.5,
 disc_path_bins=4, disc_path_eqcounts=False,
 alpha_scores=0.5, which_trees='majority', precis_threshold=0.95):

    # create a list to collect completed rule accumulators
    completed_rule_accs = [[]] * (batch_size * n_batches)

    for b in range(n_batches):
        print('walking forest for batch ' + str(b))
        instances, labels = getter.get_next(batch_size)
        instance_ids = labels.index.tolist()
        # get all the tree paths instance by instance
        walked = f_walker.forest_walk(instances = instances
                                , labels = labels)

        # rearrange paths by instances
        walked.flip()

        for i in range(batch_size):

            # process the path info for freq patt mining
            walked.set_paths(i, which_trees=which_trees)
            walked.discretize_paths(data_container.var_dict,
                                    bins=disc_path_bins,
                                    equal_counts=disc_path_eqcounts)
            # the pattens are found but not scored and sorted yet
            walked.set_patterns(support=support_paths, alpha=alpha_paths, sort=False)
            # the patterns will be weighted by chi**2 for independence test, p-values
            weights = [] * len(walked.patterns)
            for wp in walked.patterns:
                rt = rule_tester(data_container=data_container,
                rule=wp,
                sample_instances=sample_instances)
                rt.sample_instances = encoder.transform(rt.sample_instances)
                idx = rt.apply_rule()
                covered = p_count_corrected(sample_labels[idx], [i for i in range(len(data_container.class_names))])['counts']
                not_covered = p_count_corrected(sample_labels[~idx], [i for i in range(len(data_container.class_names))])['counts']
                observed = np.array((covered, not_covered))
                # weights.append(sqrt(chi2_contingency(observed=observed, correction=True)[0]))

                if covered.sum() > 0 and not_covered.sum() > 0: # previous_counts.sum() == 0 is impossible
                    weights.append(sqrt(chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)[0]))
                else:
                    weights.append(max(weights))

            # now the patterns are scored and sorted
            walked.set_patterns(support=support_paths, alpha=alpha_paths, sort=True, weights=weights) # with chi2 and support sorting
            # walked.set_patterns(support=support_paths, alpha=alpha_paths, sort=True) # with only support sorting

            # re-run the profile to greedy precis
            ra_gprec = rule_accumulator(data_container=data_container, paths_container=walked, instance_id=instance_ids[i])
            ra_gprec.profile(encoder=encoder, sample_instances=sample_instances, sample_labels=sample_labels, greedy='precision', prediction_model=f_walker.prediction_model, precis_threshold=precis_threshold)
            ra_gprec.prune_rule()
            ra_gprec_lite = ra_gprec.lite_instance()
            # del ra_gprec

            # collect completed rule accumulator
            completed_rule_accs[b * batch_size + i] = [ra_gprec_lite]

    algorithm = ['greedy_prec'] # this method proved to be the best. For alternatives, go to the github and see older versions
    return(completed_rule_accs, algorithm)

def anchors_preproc(dataset, random_state, iv_low, iv_high):
    mydata = dataset(random_state)
    tt = mydata.xval_split(iv_low=iv_low, iv_high=iv_high, test_index=random_state, random_state=123)

    # mappings for anchors
    mydata.class_names=mydata.get_label(mydata.class_col, [i for i in range(len(mydata.class_names))]).tolist()
    mydata.unsorted_categorical = [(v, mydata.var_dict[v]['order_col']) for v in mydata.var_dict if mydata.var_dict[v]['data_type'] == 'nominal' and mydata.var_dict[v]['class_col'] != True]
    mydata.categorical_features = [c[1] for c in sorted(mydata.unsorted_categorical, key = lambda x: x[1])]
    mydata.categorical_names = {i : mydata.var_dict[v]['labels'] for v, i in mydata.unsorted_categorical}

    # discretizes all cont vars
    disc = limtab.QuartileDiscretizer(data=np.array(mydata.data.drop(labels=mydata.class_col, axis=1)),
                                             categorical_features=mydata.categorical_features,
                                             feature_names=mydata.features)

    # update the tt object
    tt['X_train'] = np.array(disc.discretize(np.array(tt['X_train'])))
    tt['X_test'] = np.array(disc.discretize(np.array(tt['X_test'])))
    tt['y_train'] = np.array(tt['y_train'])
    tt['y_test'] = np.array(tt['y_test'])

    # add the mappings of discretized vars for anchors
    mydata.categorical_names.update(disc.names)

    explainer = anchtab.AnchorTabularExplainer(mydata.class_names, mydata.features, tt['X_train'], mydata.categorical_names)
    explainer.fit(tt['X_train'], tt['y_train'], tt['X_test'], tt['y_test'])
    # update the tt object
    tt['encoder'] = explainer.encoder
    tt['X_train_enc'] = explainer.encoder.transform(tt['X_train'])

    return(mydata, tt, explainer)

def anchors_explanation(instance, explainer, forest, random_state=123, threshold=0.95):
    np.random.seed(random_state)
    exp = explainer.explain_instance(instance, forest.predict, threshold=threshold)
    return(exp)