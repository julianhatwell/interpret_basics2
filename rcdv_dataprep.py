import urllib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import pickle
from itertools import chain

pickle_dir = 'rcdv_pickles'

print(
'''
Data Set Information:
This is a description of the data on the file, DATA1978.
The description was prepared by Peter Schmidt, Department of Economics, Michigan State University, East Lansing, Michigan 48824.
The data were gathered as part of a grant from the National Institute of Justice to Peter Schmidt and Ann Witte, “Improving Predictions of Recidivism by Use of Individual Characteristics,” 84-IJ-CX-0021.
A more complete description of the data, and of the uses to which they were put, can be found in the final report for this grant.
Another similar dataset, contained in a file DATA1980 on a separate diskette, is also described in that report.

The North Carolina Department of Correction furnished a data tape which was to contain information on all individuals released from a North Carolina prison during the period from July 1, 1977 through June 30, 1978.
There were 9457 individual records on this tape. However, 130 records were deleted because of obvious defects.
In almost all cases, the reason for deletion is that the individual’s date of release was in fact not during the time period which defined the data set.
This left a total of 9327 individual records, and accordingly there are 9327 records on DATA1978.

The basic sample of 9327 observations contained many observations for which one or more of the variables used in our analyses were missing.
Specifically, 4709 observations were missing information on one or more such variables, and these 4709 observations constitute the “missing data” file.
The other 4618 observations which contained complete information were randomly split into an “analysis file” of 1540 observations and a “validation file” of 3078 observations.

DATA 1978 contains 9327 individual records. Each individual record contains 28 columns of data, representing the following 19 variables.

WHITE ALCHY JUNKY SUPER MARRIED FELON WORKREL PROPTY PERSON MALE PRIORS SCHOOL RULE AGE TSERVD FOLLOW RECID TIME FILE
1 2 3 4 5 6 7 8 9 10 11-12 13-14 15-16 17-19 20-22 23-24 25-27 28

WHITE is a dummy (indicator) variable equal to zero if the individual is black, and equal to one otherwise. Basically, WHITE equals one for whites and zero for blacks. However, the North Carolina prison population also contains a small number of Native Americans, Hispanics, Orientals, and individuals of “other” race. They are treated as whites, by the above definition.
ALCHY is a dummy variable equal to one if the individual’s record indicates a serious problem with alcohol, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of ALCHY is recorded as zero, but is meaningless.
JUNKY is a dummy variable equal to one if the individual’s record indicates use of hard drugs, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of JUNKY is recorded as zero, but is meaningless.
SUPER is a dummy variable equal to one if the individual’s release from the sample sentence was supervised (e.g., parole), and equal to zero otherwise.
MARRIED is a dummy variable equal to one if the individual was married at the time of release from the sample sentence, and equal to zero otherwise.
FELON is a dummy variable equal to one if the sample conviction was for a felony, and equal to zero if it was for a misdemeanor.
WORKREL is a dummy variable equal to one if the individual participated in the North Carolina prisoner work release program during the sample sentence, and equal to zero otherwise.
PROPTY is a dummy variable equal to one if the sample conviction was for a crime against property, and equal to zero otherwise. A detailed listing of the crime codes which define this variable (and PERSON below) can be found in A. Witte, Work Release in North Carolina: An Evaluation of Its Post Release Effects, Chapel Hill, North Carolina: Institute for Research in Social Science.
PERSON is a dummy variable equal to one if the sample conviction was for a crime against a person, and equal to zero otherwise. (Incidentally, note that PROPTY plus PERSON is not necessarily equal to one, because there is an additional miscellaneous category of offenses which are neither offenses against property nor offenses against a person.)
MALE is a dummy variable equal to one if the individual is male, and equal to zero if the individual is female.
PRIORS is the number of previous incarcerations, not including the sample sentence. The value -9 indicates that this information is missing.
SCHOOL is the number of years of formal schooling completed. The value zero indicates that this information is missing.
RULE is the number of prison rule violations reported during the sample sentence.
AGE is age (in months) at time of release.
TSERVD is the time served (in months) for the sample sentence.
FOLLOW is the length of the followup period, in months. (The followup period is the time from relase until the North Carolina Department of Correction records were searched, in April, 1984.)
RECID is a dummy variable equal to one if the individual returned to a North Carolina prison during the followup period, and equal to zero otherwise.
TIME is the length of time from release from the sample sentence until return to prison in North Carolina, for individuals for whom RECID equals one. TIME is rounded to the nearest month. (In particular, note that TIME equals zero for individuals who return to prison in North Carolina within the first half month after release.) For individuals for whom RECID equals zero, the value of TIME is meaningless. For such individuals, TIME is usually recorded as zero, but it is occasionally recorded as the length of the followup period. We emphasize again that neither value is meaningful, for those individuals for whom RECID equals zero.
FILE is a variable indicating to which data sample the individual record belongs. The value 1 indicates the analysis sample, 2 the validation sampel and 3 is missing data sample.
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

rcdv = pd.read_excel(pickle_path('rcdv.xlsx')
                                , sheet_name='1978'
                                , header=0)
rcdv = rcdv.append(pd.read_excel(pickle_path('rcdv.xlsx')
                                , sheet_name='1980'
                                , header=0))

# merging two sheets brought in row numbers
rcdv.reset_index()

# remove cols we don't want. rename file to miss as it is needed to indicate where were missing values
rcdv.drop(labels='Column1', axis=1, inplace=True)
rcdv.columns = ['miss' if vn == 'file' else vn for vn in rcdv.columns]

var_names=list(rcdv)[:16] + list(rcdv)[17:] + list(rcdv)[16:17] # put recid to the end
rcdv = rcdv[var_names]

vars_types = ['nominal'
            , 'nominal'
            , 'nominal'
            , 'nominal'
            , 'nominal'
            , 'nominal'
            , 'nominal'
            , 'nominal'
            , 'nominal'
            , 'nominal'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'continuous'
            , 'nominal'
            , 'continuous'
            , 'nominal'
            , 'nominal']

class_col = 'recid'
features = [vn for vn in var_names if vn != class_col]

# recode priors, all that were set to -9 were missing, and it is logged in the file variable (3 = missing data indicator)
rcdv['priors'] = rcdv['priors'].apply(lambda x: 0 if x == -9 else x)
rcdv['miss'] = rcdv['miss'].apply(lambda x: 1 if x == 3 else 0)

# remove cols we don't want. Time is only useful in survival analysis. Correlates exactly with recid.
to_be_del = ['time']
for tbd in to_be_del:
    del rcdv[tbd]
    del vars_types[np.where(np.array(var_names) == tbd)[0][0]]
    del var_names[np.where(np.array(var_names) == tbd)[0][0]]
    del features[np.where(np.array(features) == tbd)[0][0]]

# save it out for # R
rcdv.to_csv(pickle_path('rcdv.csv.gz'), index=False, compression='gzip')

# the following creates a copy of the data frame with int mappings of categorical variables for scikit-learn
# and also a dictionary containing the label encoders/decoders for each column
rcdv_pre = pd.DataFrame.copy(rcdv)

le_dict = {}
vars_dict = {}
onehot_dict = {}

for v, t in zip(var_names, vars_types):
    if t == 'nominal':
        # create a label encoder for all categoricals
        le_dict[v] = LabelEncoder().fit(rcdv[v].unique())
        # create a dictionary of categorical names
        names = list(le_dict[v].classes_)
        # transform each categorical column
        rcdv_pre[v] = le_dict[v].transform(rcdv[v])
        # create the reverse lookup
        for n in names:
            onehot_dict[v + '_' + str(n)] = v
    else:
        rcdv_pre[v] = rcdv[v]

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
X, y = rcdv_pre[features], rcdv_pre[class_col]

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

train_priors = y_train.value_counts().sort_index()/len(y_train)
test_priors = y_test.value_counts().sort_index()/len(y_test)

# one hot encoding required for classifier
# otherwise integer vectors will be treated as ordinal
# OneHotEncoder takes an integer list as an argument to state which columns to encode
encoder = OneHotEncoder(categorical_features=categorical_features)
encoder.fit(rcdv_pre.as_matrix())
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
import the rcdv.csv file
create the pandas dataframe and prints head: rcdv
create the categorical var encoder dictionary: le_dict
create a function to get any code for a column name and label: get_code
create the dictionary of categorical values: categories
creates the list of one hot encoded variable names, onehot_features
create the list of class names: class_names
create the pandas dataframe with encoded vars: rcdv_pre
create the pandas dataframe containing all features less class: X
create the pandas series containing the class 'decision': y
create the training and test sets: X_train, y_train, X_test, y_test
evaluate the training and test set priors and print them: train_priors, test_priors
create a One Hot Encoder and encode the train set: X_train_enc
(avoids treating variables as ordinal or continuous)
pickles objects that are needed by later steps: encoder, X_train_enc, y_train
creates a closure with the location of the pickle files for easy access to the stored datasets: pickle_path()
''')

print("rcdv.head()")
print(rcdv.head())

shp, variables = rstr(rcdv)
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
