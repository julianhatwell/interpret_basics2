import urllib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import pickle
from itertools import chain

pickle_dir = 'cardiotography_pickles'

print(
'''
Data Set Information:
2126 fetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C. ...) and to a fetal state (N, S, P). Therefore the dataset can be used either for 10-class or 3-class experiments.


Attribute Information:
LB - FHR baseline (beats per minute)
AC - # of accelerations per second
FM - # of fetal movements per second
UC - # of uterine contractions per second
DL - # of light decelerations per second
DS - # of severe decelerations per second
DP - # of prolongued decelerations per second
ASTV - percentage of time with abnormal short term variability
MSTV - mean value of short term variability
ALTV - percentage of time with abnormal long term variability
MLTV - mean value of long term variability
Width - width of FHR histogram
Min - minimum of FHR histogram
Max - Maximum of FHR histogram
Nmax - # of histogram peaks
Nzeros - # of histogram zeros
Mode - histogram mode
Mean - histogram mean
Median - histogram median
Variance - histogram variance
Tendency - histogram tendency
CLASS - FHR pattern class code (1 to 10)
NSP - fetal state class code (N=normal; S=suspect; P=pathologic)
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

var_names = ['LB'
            , 'AC'
            , 'FM'
            , 'UC'
            , 'DL'
            , 'DS'
            , 'DP'
            , 'ASTV'
            , 'MSTV'
            , 'ALTV'
            , 'MLTV'
            , 'Width'
            , 'Min'
            , 'Max'
            , 'Nmax'
            , 'Nzeros'
            , 'Mode'
            , 'Mean'
            , 'Median'
            , 'Variance'
            , 'Tendency'
            , 'CLASS'
            , 'NSP']

vars_types = ['continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'nominal'
            , 'nominal']

class_col = 'NSP'
features = [vn for vn in var_names if vn != class_col]

cardiotography = pd.read_excel('CTG.xlsx'
                                , header=None
                                , names=var_names)

# re-code NSP and delete class variable
NSP = pd.Series(['N'] * cardiotography.shape[0])
NSP.loc[cardiotography.NSP.values == 2] = 'S'
NSP.loc[cardiotography.NSP.values == 3] = 'P'
cardiotography.NSP = NSP

to_be_del = ['CLASS']
for tbd in to_be_del:
    del cardiotography[tbd]
    del vars_types[np.where(np.array(var_names) == tbd)[0][0]]
    del var_names[np.where(np.array(var_names) == tbd)[0][0]]
    del features[np.where(np.array(features) == tbd)[0][0]]

# the following creates a copy of the data frame with int mappings of categorical variables for scikit-learn
# and also a dictionary containing the label encoders/decoders for each column
cardiotography_pre = pd.DataFrame.copy(cardiotography)

le_dict = {}
vars_dict = {}
onehot_dict = {}

for v, t in zip(var_names, vars_types):
    if t == 'nominal':
        # create a label encoder for all categoricals
        le_dict[v] = LabelEncoder().fit(cardiotography[v].unique())
        # create a dictionary of categorical names
        names = list(le_dict[v].classes_)
        # transform each categorical column
        cardiotography_pre[v] = le_dict[v].transform(cardiotography[v])
        # create the reverse lookup
        for n in names:
            onehot_dict[v + '_' + str(n)] = v
    else:
        cardiotography_pre[v] = cardiotography[v]

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
X, y = cardiotography_pre[features], cardiotography_pre[class_col]

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

train_priors = y_train.value_counts().sort_index()/len(y_train)
test_priors = y_test.value_counts().sort_index()/len(y_test)

# one hot encoding required for classifier
# otherwise integer vectors will be treated as ordinal
# OneHotEncoder takes an integer list as an argument to state which columns to encode
encoder = OneHotEncoder(categorical_features=categorical_features)
encoder.fit(cardiotography_pre.as_matrix())
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
import the cardiotography.csv file
create the pandas dataframe and prints head: cardiotography
create the categorical var encoder dictionary: le_dict
create a function to get any code for a column name and label: get_code
create the dictionary of categorical values: categories
creates the list of one hot encoded variable names, onehot_features
create the list of class names: class_names
create the pandas dataframe with encoded vars: cardiotography_pre
create the pandas dataframe containing all features less class: X
create the pandas series containing the class 'decision': y
create the training and test sets: X_train, y_train, X_test, y_test
evaluate the training and test set priors and print them: train_priors, test_priors
create a One Hot Encoder and encode the train set: X_train_enc
(avoids treating variables as ordinal or continuous)
pickles objects that are needed by later steps: encoder, X_train_enc, y_train
creates a closure with the location of the pickle files for easy access to the stored datasets: pickle_path()
''')

print("cardiotography.head()")
print(cardiotography.head())

shp, variables = rstr(cardiotography)
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
