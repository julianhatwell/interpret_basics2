import urllib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import pickle
from itertools import chain

from datetime import datetime
import julian
jul_conv = lambda x : 0 if str(x) == 'nan' else julian.to_jd(datetime.strptime(x, '%b-%Y'))

pickle_dir = 'lending_pickles'

print(
'''
Data Set Information:
Orignates from: https://www.lendingclub.com/info/download-data.action

See also:
https://www.kaggle.com/wordsforthewise/lending-club

Prepared by Nate George: https://github.com/nateGeorge/preprocess_lending_club_data
'''
)

# helper function for data frame str / summary
def rstr(lending):
    return lending.shape, lending.apply(lambda x: [x.unique()])

# helper function for pickling files
def pickle_path(filename):
    return(pickle_dir + '\\' + filename)

# random seed for train test split and sampling
seed = 123
class_col = 'loan_status'

if True:

    lending = pd.read_csv(pickle_path('accepted_2007_to_2017Q3.csv.gz'), compression='gzip', low_memory=True)
    # low_memory=False prevents mixed data types in the DataFrame

    # Just looking at loans that met the policy and were either fully paid or charged off (finally defaulted)
    lending = lending.loc[lending['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    # data set is wide. What can be done to reduce it?

    # drop cols with only one distinct value
    drop_list = []
    for col in lending.columns:
        if lending[col].nunique() == 1:
            drop_list.append(col)

    lending.drop(labels=drop_list, axis=1, inplace=True)

    # drop cols with excessively high missing amounts
    drop_list = []
    for col in lending.columns:
        if lending[col].notnull().sum() / lending.shape[0] < 0.5:
            drop_list.append(col)

    lending.drop(labels=drop_list, axis=1, inplace=True)

    # more noisy columns
    lending.drop(labels=['id', 'title', 'emp_title', 'application_type', 'acc_now_delinq', 'num_tl_120dpd_2m', 'num_tl_30dpd'], axis=1, inplace=True) # title is duplicated in purpose

    # convert dates to integers
    for date_col in ['issue_d', 'last_credit_pull_d', 'earliest_cr_line', 'last_pymnt_d']:
        lending[date_col] = lending[date_col].map(jul_conv)

    # highly correlated with default
    lending.drop(labels=['collection_recovery_fee', 'debt_settlement_flag', 'recoveries'], axis=1, inplace=True)

    # convert 'term' to int
    lending['term'] = lending['term'].apply(lambda s:np.float(s[1:3])) # There's an extra space in the data for some reason

    # convert sub-grade to float and remove grade
    grade_dict = {'A':0.0, 'B':1.0, 'C':2.0, 'D':3.0, 'E':4.0, 'F':5.0, 'G':6.0}
    def grade_to_float(s):
        return 5 * grade_dict[s[0]] + np.float(s[1]) - 1
    lending['sub_grade'] = lending['sub_grade'].apply(lambda s: grade_to_float(s))
    lending.drop(labels='grade', axis=1, inplace=True)

    # convert emp_length - assume missing and < 0 is no job or only very recent started job
    # emp length is only significant for values of 0 or not 0
    def emp_conv(e):
        if pd.isnull(e):
            return 'U'
        elif e[0] == '<':
            return 'U'
        else:
            return 'E'
    lending['emp'] = lending['emp_length'].apply(lambda e: emp_conv(e))
    lending.drop(labels='emp_length', axis=1, inplace=True)

    # tidy up some very minor class codes in home ownership
    lending['home_ownership'] = lending['home_ownership'].apply(lambda h: 'OTHER' if h in ['ANY', 'NONE'] else h)

    # there is a number of rows that have missing data for many variables in a block pattern -
    # these are probably useless because missingness goes across so many variables
    # it might be possible to save them to a different set and create a separate model on them

    # another approach is to fill them with an arbitrary data point (means, zeros, whatever)
    # and add a new feature that is binary for whether this row had missing data
    # this will give the model something to adjust/correlate/associate with if these rows turn out to add noise

    # 'avg_cur_bal is a template for block missingness
    # will introduce a missing indicator column based on this
    # then fillna with zeros and finally filter out some unsalvageable really rows
    lending['block_missingness'] = lending['avg_cur_bal'].isnull() * 1.0

    # this one feature has just a tiny number of zero (invalid date) and NaNs.
    # will put these in into the same state and deal with them together
    lending['last_credit_pull_d'] = lending['last_credit_pull_d'].apply(lambda x: np.nan if x == 0 else x)
    lending['last_credit_pull_d'] = lending.last_credit_pull_d.fillna(lending.last_credit_pull_d.mean())

    lending = lending.fillna(0)
    # rows where last_pymnt_d is zero are just a mess, get them outa here
    lending = lending[lending.last_pymnt_d != 0]

    # no need for an upper and lower fico, they are perfectly correlated
    fic = ['fico_range_low', 'fico_range_high']
    lastfic = ['last_fico_range_low', 'last_fico_range_high']
    lending['fico'] = lending[fic].mean(axis=1)
    lending['last_fico'] = lending[lastfic].mean(axis=1)
    lending.drop(labels=fic + lastfic, axis=1, inplace=True)

    # slightly more informative coding of these vars that are mostly correlated with loan amnt and/or high skew
    lending['non_funded_score'] = np.log(lending['loan_amnt'] + 1 - lending['funded_amnt'])
    lending['non_funded_inv_score'] = np.log(lending['loan_amnt'] + 1 - lending['funded_amnt_inv'])
    lending['adj_log_dti'] = np.log(lending['dti'] + abs(lending['dti'].min()) + 1)
    lending['log_inc'] = np.log(lending['annual_inc'] + abs(lending['annual_inc'].min()) + 1)
    lending.drop(['funded_amnt', 'funded_amnt_inv', 'dti', 'annual_inc'], axis=1, inplace=True)

    # and rearrange so class_col is at the end
    class_col = 'loan_status'
    pos = np.where(lending.columns == class_col)[0][0]
    var_names = list(lending.columns[:pos]) + list(lending.columns[pos + 1:]) + list(lending.columns[pos:pos + 1])
    lending = lending[var_names]

    # create a small set that is easier to play with on a laptop
    lend_samp = lending.sample(frac=0.1, random_state=seed).reset_index()
    lend_samp.drop(labels='index', axis=1, inplace=True)

    # full set
    lending.to_csv(pickle_path('lending.csv.gz'), index=False, compression='gzip')
    lend_samp.to_csv(pickle_path('lend_samp.csv.gz'), index=False, compression='gzip')


### missing data analysis was done to get to the above strategy
# import missingno as msno
# missingdata_df = lending.columns[lending.isnull().any()].tolist()
# msno.matrix(lending[missingdata_df])
# msno.bar(lending[missingdata_df], color="blue", log=False, figsize=(30,18))
# msno.heatmap(lending[missingdata_df], figsize=(20,20))
# msno.dendrogram(lending[missingdata_df], figsize=(20,20))

lending = pd.read_csv(pickle_path('lend_samp.csv.gz'), compression='gzip')
var_names = lending.columns
vars_types = ['nominal' if dt.name == 'object' else 'continuous' for dt in lending.dtypes.values]
features = [vn for vn in var_names if vn != class_col]

# the following creates a copy of the data frame with int mappings of categorical variables for scikit-learn
# and also a dictionary containing the label encoders/decoders for each column
lending_pre = pd.DataFrame.copy(lending)

le_dict = {}
vars_dict = {}
onehot_dict = {}

for v, t in zip(var_names, vars_types):
    if t == 'nominal':
        # create a label encoder for all categoricals
        le_dict[v] = LabelEncoder().fit(lending[v].unique())
        # create a dictionary of categorical names
        names = list(le_dict[v].classes_)
        # transform each categorical column
        lending_pre[v] = le_dict[v].transform(lending[v])
        # create the reverse lookup
        for n in names:
            onehot_dict[v + '_' + str(n)] = v
    else:
        lending_pre[v] = lending[v]

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
X, y = lending_pre[features], lending_pre[class_col]

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

train_priors = y_train.value_counts().sort_index()/len(y_train)
test_priors = y_test.value_counts().sort_index()/len(y_test)

# one hot encoding required for classifier
# otherwise integer vectors will be treated as ordinal
# OneHotEncoder takes an integer list as an argument to state which columns to encode
encoder = OneHotEncoder(categorical_features=categorical_features)
encoder.fit(lending_pre.as_matrix())
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
import the lending.csv file
create the pandas dataframe and prints head: lending
create the categorical var encoder dictionary: le_dict
create a function to get any code for a column name and label: get_code
create the dictionary of categorical values: categories
creates the list of one hot encoded variable names, onehot_features
create the list of class names: class_names
create the pandas dataframe with encoded vars: lending_pre
create the pandas dataframe containing all features less class: X
create the pandas series containing the class 'decision': y
create the training and test sets: X_train, y_train, X_test, y_test
evaluate the training and test set priors and print them: train_priors, test_priors
create a One Hot Encoder and encode the train set: X_train_enc
(avoids treating variables as ordinal or continuous)
pickles objects that are needed by later steps: encoder, X_train_enc, y_train
creates a closure with the location of the pickle files for easy access to the stored datasets: pickle_path()
''')

print("lending.head()")
print(lending.head())

shp, variables = rstr(lending)
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
