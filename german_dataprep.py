import urllib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import pickle
from itertools import chain

pickle_dir = 'german_pickles'

print(
'''
Source:
Professor Dr. Hans Hofmann
Institut f"ur Statistik und "Okonometrie
Universit"at Hamburg
FB Wirtschaftswissenschaften
Von-Melle-Park 5
2000 Hamburg 13

Data Set Information:
Two datasets are provided. the original dataset, in the form provided by Prof. Hofmann, contains categorical/symbolic attributes and is in the file "german.data".
For algorithms that need numerical attributes, Strathclyde University produced the file "german.data-numeric". This file has been edited and several indicator variables added to make it suitable for algorithms which cannot cope with categorical variables. Several attributes that are ordered categorical (such as attribute 17) have been coded as integer. This was the form used by StatLog.

This dataset requires use of a cost matrix:
. 1 2
------
1 0 1
-----
2 5 0

(1 = Good, 2 = Bad)
The rows represent the actual classification and the columns the predicted classification.
It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).
'''
)

# helper function for data frame str / summary
def rstr(df):
    return df.shape, df.apply(lambda x: [x.unique()])

# helper function for pickling files
def pickle_path(filename):
    return(pickle_dir + '\\' + filename)

# random seed for test_train_split
seed=123

var_names = ['chk'
            , 'dur'
            , 'crhis'
            , 'pps'
            , 'amt'
            , 'svng'
            , 'emp'
            , 'rate'
            , 'pers'
            , 'debt'
            , 'res'
            , 'prop'
            , 'age'
            , 'plans'
            , 'hous'
            , 'creds'
            , 'job'
            , 'deps'
            , 'tel'
            , 'foreign'
            , 'rating']

vars_types = ['nominal'
            , 'continuous'
            , 'nominal'
            , 'nominal'
            , 'continuous'
            , 'nominal'
            , 'nominal'
            , 'continuous'
            , 'nominal'
            , 'nominal'
            , 'continuous'
            , 'nominal'
            , 'continuous'
            , 'nominal'
            , 'nominal'
            , 'continuous'
            , 'nominal'
            , 'continuous'
            , 'nominal'
            , 'nominal'
            , 'nominal']

class_col = 'rating'
features = [vn for vn in var_names if vn != class_col]

if True:
    '''
    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

    german_bytes = urllib.request.urlopen(target_url)
    german = pd.read_csv(german_bytes,
                         header=None,
                         delimiter=' ',
                         index_col=False,
                         names=var_names)

    # re-code rating class variable
    rating = pd.Series(['good'] * german.count()[0])
    rating.loc[german.rating == 2] = 'bad'
    german.rating = rating

    # kill continuous vars for now
    # to_be_del = ['dur', 'amt', 'rate', 'res', 'age', 'creds', 'deps']
    #for tbd in to_be_del:
    #    del german[tbd]
    #    del vars_types[np.where(np.array(var_names) == tbd)[0][0]]
    #    del var_names[np.where(np.array(var_names) == tbd)[0][0]]
    #    del features[np.where(np.array(features) == tbd)[0][0]]

    german.to_csv(pickle_path('german.csv.gz'), index=False, compression='gzip')
    '''
    
german = pd.read_csv(pickle_path('german.csv.gz'), compression='gzip')

# the following creates a copy of the data frame with int mappings of categorical variables for scikit-learn
# and also a dictionary containing the label encoders/decoders for each column
german_pre = pd.DataFrame.copy(german)

le_dict = {}
vars_dict = {}
onehot_dict = {}

for v, t in zip(var_names, vars_types):
    if t == 'nominal':
        # create a label encoder for all categoricals
        le_dict[v] = LabelEncoder().fit(german[v].unique())
        # create a dictionary of categorical names
        names = list(le_dict[v].classes_)
        # transform each categorical column
        german_pre[v] = le_dict[v].transform(german[v])
        # create the reverse lookup
        for n in names:
            onehot_dict[v + '_' + str(n)] = v
    else:
        german_pre[v] = german[v]

    vars_dict[v] = {'labels' : names if t == 'nominal' else None
                    , 'onehot_labels' : [v + '_' + str(n) for n in names] if t == 'nominal' else None
                    , 'class_col' : True if v == class_col else False
                    , 'data_type' : t}

categorical_features=[i for i, (c, t) in enumerate(zip([vars_dict[f]['class_col'] for f in features],
[vars_dict[f]['data_type'] == 'nominal' for f in features])) if not c and t]

# creates a flat list just for the features
onehot_features = []
continuous_features = []
for f, t in zip(var_names, vars_types):
    if f == class_col: continue
    if t == 'continuous':
        continuous_features.append(f)
    else:
        onehot_features.append(vars_dict[f]['onehot_labels'])

# They get stuck on the end by encoding
onehot_features.append(continuous_features)
# flatten out the nesting
onehot_features = list(chain.from_iterable(onehot_features))

# a function to return any code from a label
def get_code(col, label):
    return le_dict[col].transform([label])[0]

# a function to return any label from a code
def get_label(col, label):
    return le_dict[col].inverse_transform([label])[0]

# there is a bug in sklearn causing all the warnings. This should be fixed in next release.
def pretty_print_tree_votes(paths, preds, labels):
    for instance in paths.keys():
        print('Instance ' + str(instance) + ':    True Class = ' +
        str(labels.values[instance]) + ' ' +
          str(get_label(class_col, labels.values[instance])) +
          '    Pred Class = ' + str(preds[instance]) + ' ' +
          str(get_label(class_col, preds[instance])) +
          '    Majority voting trees = ' + str(len(paths[instance])))

class_names = list(le_dict[class_col].classes_)

# train test splitting
X, y = german_pre[features], german_pre[class_col]

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

train_priors = y_train.value_counts().sort_index()/len(y_train)
test_priors = y_test.value_counts().sort_index()/len(y_test)

# one hot encoding required for classifier
# otherwise integer vectors will be treated as ordinal
# OneHotEncoder takes an integer list as an argument to state which columns to encode
encoder = OneHotEncoder(categorical_features=categorical_features)
encoder.fit(german_pre.as_matrix())
X_train_enc = encoder.transform(X_train)

if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)

encoder_store = open(pickle_path('encoder.pickle'), "wb")
pickle.dump(encoder, encoder_store)
encoder_store.close()

X_train_enc_store = open(pickle_path('X_train_enc.pickle'), "wb")
pickle.dump(X_train_enc, X_train_enc_store)
X_train_enc_store.close()

y_train_store = open(pickle_path('y_train.pickle'), "wb")
pickle.dump(y_train, y_train_store)
y_train_store.close()

pickle_dir_store = open("pickle_dir.pickle", "wb")
pickle.dump(pickle_dir, pickle_dir_store)
pickle_dir_store.close()

print('''Utility code in the associated file performs the following steps:
set random seed for the test_train_split
import packages and modules
defines a custom summary function: rstr()
create the list of variable names: var_names
create the list of features (var_names less class): features
import the german.csv file
create the pandas dataframe and prints head: german
create the categorical var encoder dictionary: le_dict
create a function to get any code for a column name and label: get_code
create the dictionary of categorical values: categories
creates the list of one hot encoded variable names, onehot_features
create the list of class names: class_names
create the pandas dataframe with encoded vars: german_pre
create the pandas dataframe containing all features less class: X
create the pandas series containing the class 'decision': y
create the training and test sets: X_train, y_train, X_test, y_test
evaluate the training and test set priors and print them: train_priors, test_priors
create a One Hot Encoder and encode the train set: X_train_enc
(avoids treating variables as ordinal or continuous)
pickles objects that are needed by later steps: encoder, X_train_enc, y_train
creates a closure with the location of the pickle files for easy access to the stored datasets: pickle_path()
''')

print("german.head()")
print(german.head())

shp, variables = rstr(german)
print()
print("shape")
print(shp)
print()
print("variables summary")
print(variables)

print("\n")
print("Training Priors")
for c, p in zip(class_names, train_priors):
        print(c, p)

print("\n")
print("Test Priors")
for c, p in zip(class_names, test_priors):
        print(c, p)
