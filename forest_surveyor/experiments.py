import time
import timeit

import numpy as np
from pandas import DataFrame, Series

import multiprocessing as mp
from forest_surveyor.mp_callable import mp_experiment

# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def grid_experiment_mp(grid):
    # capture timing results
    start_time = timeit.default_timer()

    print(str(len(grid.index)) + ' runs to do')

    n_cores = mp.cpu_count()-1
    print('Going parallel over ' + str(n_cores) + ' cores ... (please wait)')
    pool = mp.Pool(processes=n_cores)
    results = []

    # iterate over the paramaters for each run
    for g in range(len(grid.index)):
        # ugly code because args must be ordered tuple, no keywords are working
        grid_idx = grid.loc[g].grid_idx
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
        # results are just the elapsed times and grid ids. The output objects have been saved to file as a side effect
        results.append(pool.apply_async(mp_experiment, (grid_idx, dataset, random_state, add_trees,
                                                                override_tuning, n_instances, n_batches,
                                                                eval_model, alpha_scores, alpha_paths,
                                                                support_paths, precis_threshold, run_anchors,
                                                                which_trees, disc_path_bins, disc_path_eqcounts,
                                                                iv_low, iv_high)
                                                                ))
    # clean up the multiprocessing
    pool.close()
    pool.join()

    print('Completed ' + str(len(grid)) + ' run(s) and exited parallel')
    print()
    # capture timing results
    elapsed = timeit.default_timer() - start_time
    print('Time elapsed:', "{:0.4f}".format(elapsed), 'seconds')

    return(results)
