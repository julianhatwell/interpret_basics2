import json
import time
import timeit
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterGrid
import sklearn.metrics as metrics

# get the correct directory for saved objects
pickle_dir_store = open("pickle_dir.pickle", "rb")
pickle_dir = pickle.load(pickle_dir_store)
pickle_dir_store.close()

# helper function for pickling files
def pickle_path(filename):
    return(pickle_dir + '\\' + filename)

# load up the training set (required because of running from script into Jup Note)
encoder_store = open(pickle_path('encoder.pickle'), "rb")
encoder = pickle.load(encoder_store)
encoder_store.close()

X_train_enc_store = open(pickle_path('X_train_enc.pickle'), "rb")
X_train_enc = pickle.load(X_train_enc_store)
X_train_enc_store.close()

y_train_store = open(pickle_path('y_train.pickle'), "rb")
y_train = pickle.load(y_train_store)
y_train_store.close()

# random seed for random forest
seed=123

#################################################
### Won't run this again as it takes too long ###
#################################################

if True:
    '''
    import json
    import timeit
    from sklearn.ensemble import RandomForestClassifier

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
        rf.set_params(oob_score = True, random_state=seed, **g)
        rf.fit(X_train_enc, y_train)
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

    with open(pickle_path('params.json'), 'w') as outfile:
        json.dump(params, outfile)

    params = pd.DataFrame(params)
    '''

print('''Parameter tuning (disabled)
Utility code in the associated file performs the following steps:
set random seed for the random forest
fetch the best parameters from model tuning results
''')

# get the params from a local text file
# TO DO: a better file name for generic process
with open(pickle_path('params.json'), 'r') as infile:
    params = json.load(infile)
params = pd.DataFrame(params)

# get the best params
best_grid = params.loc[params['score'].idxmax()]
best_params = {k: int(v) for k, v in best_grid.items() if k not in ('score', 'elapsed_time')}
print("Best OOB Cohen's Kappa during tuning: " "{:0.4f}".format(best_grid.score))
print("Best parameters:", best_params)
print()

print("Training a random forest model using best parameters... (please wait)")
rf = RandomForestClassifier(random_state=seed, oob_score=True, **best_params)
rf.fit(X_train_enc, y_train)
print()
print("Done")
print()

enc_model = make_pipeline(encoder, rf)
print("Created helper function enc_model(). A pipeline: feature encoding -> rf model")
