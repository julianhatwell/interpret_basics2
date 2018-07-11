import json
import time
import timeit
import pickle
import numpy as np
import multiprocessing as mp
from pandas import DataFrame, Series
from forest_surveyor import p_count, p_count_corrected
from forest_surveyor.plotting import plot_confusion_matrix
from forest_surveyor.structures import rule_accumulator, forest_walker, batch_getter, rule_tester, loo_encoder
from forest_surveyor.mp_callable import mp_experiment, mp_experiment_scratch
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

    elapsed = timeit.default_timer() - start_time
    print('Time elapsed:', "{:0.4f}".format(elapsed), 'seconds')

    params = DataFrame(params).sort_values(['score','n_estimators','max_depth','min_samples_leaf'],
                                        ascending=[False, True, True, False])

    best_grid = params.loc[params['score'].idxmax()]
    best_params = {k: int(v) if k not in ('score', 'elapsed_time') else v for k, v in best_grid.items()}

    if save_path is not None:
        with open(save_path + 'best_params.json', 'w') as outfile:
            json.dump(best_params, outfile)

    return(best_params)

def tune_rf(X, y, grid = None, random_state=123, save_path = None, override_tuning=False):

    # to do - test allowable structure of grid input
    if override_tuning:
        best_params = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)
    else:
        try:
            with open(save_path + 'best_params.json', 'r') as infile:
                best_params = json.load(infile)
            print('Using existing params file. To re-tune, delete file at ' + save_path + 'best_params.json')
            print()
            return(best_params)
        except:
            best_params = do_tuning(X, y, grid=grid, random_state=random_state, save_path=save_path)

    print("Best OOB Accuracy Estimate during tuning: " "{:0.4f}".format(best_params['score']))
    print("Best parameters:", best_params)
    print()

    return(best_params)

def train_rf(X, y, best_params = None, encoder = None, random_state = 123):

    # to do - test allowable structure of grid input
    if best_params is not None:
        best_params = {k: v for k, v in best_params.items() if k not in ('score', 'elapsed_time')}
        print("Training a random forest model using given parameters " + str(best_params) + "... (please wait)")
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

    # view the confusion matrix
    cm = confusion_matrix(y, pred)
    prfs = precision_recall_fscore_support(y, pred)
    acc = accuracy_score(y, pred)
    coka = cohen_kappa_score(y, pred)
    print("Accuracy on unseen instances: " "{:0.4f}".format(acc))
    print("Cohen's Kappa on unseen instances: " "{:0.4f}".format(coka))
    print()
    print('Precision: ' + str(prfs[0].round(2).tolist()))
    print('Recall: ' + str(prfs[1].round(2).tolist()))
    print('F1 Score: ' + str(prfs[2].round(2).tolist()))
    print('Support: ' + str(prfs[3].round(2).tolist()))
    print()

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
 data_container, encoder, sample_instances, sample_labels,
 batch_size = 1, n_batches = 1,
 support_paths=0.1, alpha_paths=0.5,
 disc_path_bins=4, disc_path_eqcounts=False,
 alpha_scores=0.5, which_trees='majority', precis_threshold=0.95):
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

            # collect results
            results[b * batch_size + i] = [ra_gprec_lite]

        # report progress
        print('done batch ' + str(b))

    result_sets = ['greedy_prec']
    return(results, result_sets)

def anchors_preproc(get_data, random_state):
    mydata = get_data(random_state)
    tt = mydata.tt_split()

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

def experiment(get_dataset, n_instances, n_batches,
 random_state=123,
 override_tuning=False,
 support_paths=0.1,
 alpha_paths=0.5,
 disc_path_bins=4,
 disc_path_eqcounts=False,
 alpha_scores=0.5,
 which_trees='majority',
 precis_threshold=0.95,
 add_trees=0,
 eval_model=False,
 run_anchors=True):

    print('LOADING NEW DATA SET.')
    print()
    # load a data set
    mydata = get_dataset(random_state=random_state)

    # train test split
    tt = mydata.tt_split()

    ################ PARAMETER TUNING ###################
    ############ Only runs when required ################
    #####################################################

    print('Finding best parameters for Random Forest. Checking for prior tuning parameters.')
    print()
    best_params = tune_rf(tt['X_train_enc'], tt['y_train'],
     save_path = mydata.pickle_path(),
     random_state=mydata.random_state,
     override_tuning=override_tuning)

    #####################################################

    # update best params according to expermental design
    best_params['n_estimators'] = best_params['n_estimators'] + add_trees

    # train a rf model
    rf, enc_rf = train_rf(X=tt['X_train_enc'], y=tt['y_train'],
     best_params=best_params,
     encoder=tt['encoder'],
     random_state=mydata.random_state)

    if eval_model:
        cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
                     class_names=mydata.get_label(mydata.class_col, [i for i in range(len(mydata.class_names))]).tolist(),
                     plot_cm=True, plot_cm_norm=True)
    else:
        cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
                     class_names=mydata.get_label(mydata.class_col, [i for i in range(len(mydata.class_names))]).tolist(),
                     plot_cm=False, plot_cm_norm=False)

    # fit the forest_walker
    f_walker = forest_walker(forest = rf,
     data_container=mydata,
     encoder=tt['encoder'],
     prediction_model=enc_rf)

    # run the batch based forest walker
    getter = batch_getter(instances=tt['X_test'], labels=tt['y_test'])

    # faster to do one batch, avoids the overhead of setting up many but consumes more mem
    # get a round number of instances no more than what's available in the test set
    n_instances = min(n_instances, len(tt['y_test']))
    batch_size = int(n_instances / n_batches)
    n_instances = batch_size * n_batches

    print('''NOTE: During run, true divide errors are acceptable.
    Returned when a tree does not contain any node for either/both upper and lower bounds of a feature.

    ''')
    print('Starting new run at: ' + time.asctime(time.gmtime()) + ' with batch_size = ' + str(batch_size) + ' and n_batches = ' + str(n_batches) + '...(please wait)')
    wb_start_time = timeit.default_timer()

    # results are for the whole batch
    results, result_sets = run_batches(f_walker=f_walker,
     getter=getter,
     data_container=mydata,
     encoder=tt['encoder'],
     sample_instances=tt['X_train'],
     sample_labels=tt['y_train'],
     support_paths=support_paths,
     alpha_paths=alpha_paths,
     disc_path_bins=disc_path_bins,
     disc_path_eqcounts=disc_path_eqcounts,
     alpha_scores=alpha_scores,
     which_trees=which_trees,
     precis_threshold=precis_threshold,
     batch_size=batch_size,
     n_batches=n_batches)

    wb_end_time = timeit.default_timer()
    wb_elapsed_time = wb_end_time - wb_start_time
    print('Done. Completed run at: ' + time.asctime(time.gmtime()) + '. Elapsed time (seconds) = ' + str(wb_elapsed_time))
    print()
    print('Compiling Training Results at: ' + time.asctime(time.gmtime()) + '...(please wait)')
    wbres_start_time = timeit.default_timer()

    headers = ['instance_id', 'result_set',
                'pretty rule', 'rule length',
                'pred class', 'pred class label',
                'target class', 'target class label',
                'majority vote share', 'pred prior',
                'precision(tr)', 'recall(tr)', 'f1(tr)',
                'accuracy(tr)', 'lift(tr)',
                'total coverage(tr)',
                'precision(tt)', 'recall(tt)', 'f1(tt)',
                'accuracy(tt)', 'lift(tt)',
                'total coverage(tt)', 'model_acc', 'model_ck']
    output = [[]] * len(results) * len(result_sets)

    # get all the label predictions done
    predictor = rule_tester(data_container=mydata,
                            rule=[], # dummy rule, not relevant
                            sample_instances=tt['X_test'])
    pred_instances, pred_labels = predictor.encode_pred(prediction_model=enc_rf, bootstrap=False)
    # leave one out encoder for test set evaluation
    looe = loo_encoder(pred_instances, pred_labels, tt['encoder'])
    for i in range(len(results)):
        # these are the same for a whole result set
        instance_id = results[i][0].instance_id
        mc = results[i][0].major_class
        mc_lab = results[i][0].major_class_label
        tc = results[i][0].target_class
        tc_lab = results[i][0].target_class_label
        mvs = results[i][0].model_post[tc]
        prior = results[i][0].pri_and_post[0][tc]
        for j, rs in enumerate(result_sets):
            rule = results[i][j].pruned_rule
            pretty_rule = mydata.pretty_rule(rule)
            rule_len = len(rule)
            tr_prec = list(reversed(results[i][j].pri_and_post))[0][tc]
            tr_recall = list(reversed(results[i][j].pri_and_post_recall))[0][tc]
            tr_f1 = list(reversed(results[i][j].pri_and_post_f1))[0][tc]
            tr_acc = list(reversed(results[i][j].pri_and_post_accuracy))[0][tc]
            tr_lift = list(reversed(results[i][j].pri_and_post_lift))[0][tc]
            tr_coverage = list(reversed(results[i][j].coverage))[0]

            # get test sample ready by leave-one-out then evaluating
            instances, enc_instances, labels = looe.loo_encode(instance_id)
            rt = rule_tester(data_container=mydata, rule=rule,
                                sample_instances=enc_instances,
                                sample_labels=labels)
            eval_rule = rt.evaluate_rule()

            # collect results
            tt_prec = eval_rule['post'][tc]
            tt_recall = eval_rule['recall'][tc]
            tt_f1 = eval_rule['f1'][tc]
            tt_acc = eval_rule['accuracy'][tc]
            tt_lift = eval_rule['lift'][tc]
            tt_coverage = eval_rule['coverage']

            output[j * len(results) + i] = [instance_id,
                    rs,
                    pretty_rule,
                    rule_len,
                    mc,
                    mc_lab,
                    tc,
                    tc_lab,
                    mvs,
                    prior,
                    tr_prec,
                    tr_recall,
                    tr_f1,
                    tr_acc,
                    tr_lift,
                    tr_coverage,
                    tt_prec,
                    tt_recall,
                    tt_f1,
                    tt_acc,
                    tt_lift,
                    tt_coverage,
                    acc,
                    coka]

    wbres_end_time = timeit.default_timer()
    wbres_elapsed_time = wbres_end_time - wbres_start_time
    print('Results Completed at: ' + time.asctime(time.gmtime()) + '. Elapsed time (seconds) = ' + str(wbres_elapsed_time))

    print('Whiteboxing Randfor Done')
    print()

    # run anchors if requested
    anch_elapsed_time = None
    if run_anchors:
        print('Processing Anchors')
        print('Starting new run at: ' + time.asctime(time.gmtime()))
        anch_start_time = timeit.default_timer()
        print()
        instance_ids = tt['X_test'].index.tolist() # record of row indices will be lost after preproc
        mydata, tt, explainer = anchors_preproc(get_dataset, random_state)

        rf, enc_rf = train_rf(tt['X_train_enc'], y=tt['y_train'],
        best_params=best_params,
        encoder=tt['encoder'],
        random_state=mydata.random_state)

        if eval_model:
            cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
                         class_names=mydata.class_names,
                         plot_cm=True, plot_cm_norm=True)
        else:
            cm, acc, coka, prfs = evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'],
                         class_names=mydata.class_names,
                         plot_cm=False, plot_cm_norm=False)
        output_anch = [[]] * n_instances
        for i in range(n_instances):
            instance_id = instance_ids[i]
            if i % 10 == 0: print('Working on Anchors for instance ' + str(instance_id))
            instance = tt['X_test'][i]
            exp = anchors_explanation(instance, explainer, rf, threshold=precis_threshold)
            # capture the explainer
            results[i].append(exp)

            # Get test examples where the anchor applies
            fit_anchor_train = np.where(np.all(tt['X_train'][:, exp.features()] == instance[exp.features()], axis=1))[0]
            fit_anchor_test = np.where(np.all(tt['X_test'][:, exp.features()] == instance[exp.features()], axis=1))[0]
            fit_anchor_test = [fat for fat in fit_anchor_test if fat != i] # exclude current instance

            # train
            priors = p_count_corrected(tt['y_train'], [i for i in range(len(mydata.class_names))])
            if any(fit_anchor_train):
                p_counts = p_count_corrected(enc_rf.predict(tt['X_train'][fit_anchor_train]), [i for i in range(len(mydata.class_names))])
            else:
                p_counts = p_count_corrected([None], [i for i in range(len(mydata.class_names))])
            counts = p_counts['counts']
            labels = p_counts['labels']
            post = p_counts['p_counts']
            p_corrected = np.array([p if p > 0.0 else 1.0 for p in post])
            cover = counts.sum() / priors['counts'].sum()
            recall = counts/priors['counts'] # recall
            r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
            observed = np.array((counts, priors['counts']))
            if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
                chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
            else:
                chisq = np.nan
            f1 = [2] * ((post * recall) / (p_corrected + r_corrected))
            not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
            accu = not_covered_counts/priors['counts'].sum()
            # to avoid div by zeros
            pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']])
            pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)])
            if counts.sum() == 0:
                rec_corrected = np.array([0.0] * len(pos_corrected))
                cov_corrected = np.array([1.0] * len(pos_corrected))
            else:
                rec_corrected = counts / counts.sum()
                cov_corrected = np.array([counts.sum() / priors['counts'].sum()])

            lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )

            # capture train
            mc = enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]
            mc_lab = mydata.class_names[enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]]
            tc = enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]
            tc_lab = mydata.class_names[enc_rf.predict(tt['X_test'][i].reshape(1, -1))[0]]
            mvs = np.nan
            prior = priors['p_counts'][tc]
            pretty_rule = ' AND '.join(exp.names())
            rule_len = len(exp.names())
            tr_prec = post[tc]
            tr_recall = recall[tc]
            tr_f1 = f1[tc]
            tr_acc = accu[tc]
            tr_lift = lift[tc]
            tr_coverage = cover

            # test
            priors = p_count_corrected(tt['y_test'], [i for i in range(len(mydata.class_names))])
            if any(fit_anchor_test):
                p_counts = p_count_corrected(enc_rf.predict(tt['X_test'][fit_anchor_test]), [i for i in range(len(mydata.class_names))])
            else:
                p_counts = p_count_corrected([None], [i for i in range(len(mydata.class_names))])
            counts = p_counts['counts']
            labels = p_counts['labels']
            post = p_counts['p_counts']
            p_corrected = np.array([p if p > 0.0 else 1.0 for p in post])
            cover = counts.sum() / priors['counts'].sum()
            recall = counts/priors['counts'] # recall
            r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
            observed = np.array((counts, priors['counts']))
            if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
                chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
            else:
                chisq = np.nan
            f1 = [2] * ((post * recall) / (p_corrected + r_corrected))
            not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
            # accuracy = (TP + TN) / num_instances formula: https://books.google.co.uk/books?id=ubzZDQAAQBAJ&pg=PR75&lpg=PR75&dq=rule+precision+and+coverage&source=bl&ots=Aa4Gj7fh5g&sig=6OsF3y4Kyk9KlN08OPQfkZCuZOc&hl=en&sa=X&ved=0ahUKEwjM06aW2brZAhWCIsAKHY5sA4kQ6AEIUjAE#v=onepage&q=rule%20precision%20and%20coverage&f=false
            accu = not_covered_counts/priors['counts'].sum()
            pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']]) # to avoid div by zeros
            pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)]) # to avoid div by zeros
            if counts.sum() == 0:
                rec_corrected = np.array([0.0] * len(pos_corrected))
                cov_corrected = np.array([1.0] * len(pos_corrected))
            else:
                rec_corrected = counts / counts.sum()
                cov_corrected = np.array([counts.sum() / priors['counts'].sum()])

            lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )

            # capture test
            tt_prec = post[tc]
            tt_recall = recall[tc]
            tt_f1 = f1[tc]
            tt_acc = accu[tc]
            tt_lift = lift[tc]
            tt_coverage = cover

            output_anch[i] = [instance_id,
                                'anchors', # result_set
                                pretty_rule,
                                rule_len,
                                mc,
                                mc_lab,
                                tc,
                                tc_lab,
                                mvs,
                                prior,
                                tr_prec,
                                tr_recall,
                                tr_f1,
                                tr_acc,
                                tr_lift,
                                tr_coverage,
                                tt_prec,
                                tt_recall,
                                tt_f1,
                                tt_acc,
                                tt_lift,
                                tt_coverage,
                                acc,
                                coka]

        output = np.concatenate((output, output_anch), axis=0)
        anch_end_time = timeit.default_timer()
        anch_elapsed_time = anch_end_time - anch_start_time
        print('Anchors Done. Completed run at: ' + time.asctime(time.gmtime()) + '. Elapsed time (seconds) = ' + str(anch_elapsed_time))

    print()
    output_df = DataFrame(output, columns=headers)
    output_df.to_csv(mydata.pickle_path(mydata.pickle_dir.replace('pickles', 'results') + '_rnst_' + str(random_state) + "_addt_" + str(add_trees) + '_timetest.csv'))
    # save the full results object
    results_store = open(mydata.pickle_path('results' + '_rnst_' + str(random_state) + "_addt_" + str(add_trees) + '.pickle'), "wb")
    pickle.dump(results, results_store)
    results_store.close()


    print('Results saved at ' + mydata.pickle_path('results' + '_rnst_' + str(random_state) + '.pickle'))
    print()
    print('To retrieve results execute the following:')
    print('results_store = open(\'' + mydata.pickle_path('results' + '_rnst_' + str(random_state) + '.pickle') + '\', "rb")')
    print('results = pickle.load(results_store)')
    print('results_store.close()')
    print()
    return(results, output_df, wb_elapsed_time + wbres_elapsed_time, anch_elapsed_time)

def grid_experiment(dsets,
                exp_grid):

    headers = ['gr_index', 'dataset_set',
                'wb_elapsed_time', 'anch_elapsed_time']
    output = [[]] * len(dsets) * len(exp_grid.index)

    for eg in exp_grid.index:
        for i, dataset in enumerate(dsets):
            results, output_df, wb_elapsed_time, anch_elapsed_time = experiment(dataset,
                            random_state=exp_grid.loc[eg]['random_state'],
                            n_instances=exp_grid.loc[eg]['n_instances'],
                            n_batches=exp_grid.loc[eg]['n_batches'],
                            eval_model=exp_grid.loc[eg]['eval_model'],
                            override_tuning=exp_grid.loc[eg]['override_tuning'],
                            support_paths=exp_grid.loc[eg]['support_paths'],
                            alpha_paths=exp_grid.loc[eg]['alpha_paths'],
                            alpha_scores=exp_grid.loc[eg]['alpha_scores'],
                            precis_threshold=exp_grid.loc[eg]['precis_threshold'],
                            add_trees=exp_grid.loc[eg]['add_trees'],
                            run_anchors=exp_grid.loc[eg]['run_anchors'],
                            which_trees=exp_grid.loc[eg]['which_trees'])

            output[eg * len(dsets) + i] = [eg,
                                            dataset.__name__,
                                            wb_elapsed_time,
                                            anch_elapsed_time]
    output_df = DataFrame(output, columns=headers)
    output_df.to_csv('whiteboxing/timetest.csv')


def grid_experiment_mp(grid):
    # capture timing results
    headers = ['gr_index', 'wb_elapsed_time', 'anch_elapsed_time']
    output = [[]] * len(grid.index)

    print('Going parallel...')
    pool = mp.Pool(processes=mp.cpu_count()-1)
    results = []
    for g in range(len(grid.index)):
        index = grid.loc[g].index
        dataset = grid.loc[g].dataset
        random_state = grid.loc[g].random_state
        add_trees = grid.loc[g].add_trees
        override_tuning = grid.loc[g].override_tuning
        n_instances = grid.loc[g].n_instances
        n_batches = grid.loc[g].n_batches
        eval_model = grid.loc[g].eval_model
        alpha_scores = grid.loc[g].alpha_scores
        alpha_paths = grid.loc[g].alpha_paths
        support_paths = grid.loc[g].support_paths
        precis_threshold = grid.loc[g].precis_threshold
        run_anchors = grid.loc[g].run_anchors
        which_trees = grid.loc[g].which_trees
        disc_path_bins = grid.loc[g].disc_path_bins
        disc_path_eqcounts = grid.loc[g].disc_path_eqcounts
        iv_low = grid.loc[g].iv_low
        iv_high = grid.loc[g].iv_high
        # ugly code because args must be ordered tuple, no keywords are working
        results.append(pool.apply_async(mp_experiment_scratch, (index, dataset, random_state, add_trees,
                                                                override_tuning, n_instances, n_batches,
                                                                eval_model, alpha_scores, alpha_paths,
                                                                support_paths, precis_threshold, run_anchors,
                                                                which_trees, disc_path_bins, disc_path_eqcounts,
                                                                iv_low, iv_high)
                                                                ))
    pool.close()
    pool.join()

    start_time = timeit.default_timer()

    # # Define an output queue
    # output = mp.Queue()
    #
    # # List of processes that we want to run
    # processes = [mp.Process(target=mp_experiment, args=(grid_line = grid.loc[g],)) for g in grid.index]
    #
    # print(str(len(grid)) + ' runs to do')
    # print()


    # # Run processes
    # for p in processes:
    #     p.start()
    # #
    # # Exit the completed processes
    # for p in processes:
    #     p.join()

    # # Get process results from the output queue
    # results_tuples = [output.get() for p in processes]

    # print('Completed ' + str(len(grid)) + ' run(s) and exited parallel')
    # print()
    # elapsed = timeit.default_timer() - start_time
    # print('Time elapsed:', "{:0.4f}".format(elapsed), 'seconds')

    return(results)
