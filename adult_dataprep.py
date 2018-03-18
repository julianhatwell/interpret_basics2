import os
import io
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import pickle
from itertools import chain
import requests

pickle_dir = 'adult_pickles'

print('''
Data Description:
This data was extracted from the adult bureau database found at
http://www.adult.gov/ftp/pub/DES/www/welcome.html
Donor: Ronny Kohavi and Barry Becker,
      Data Mining and Visualization
      Silicon Graphics.
      e-mail: ronnyk@sgi.com for questions.
Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
48842 instances, mix of continuous and discrete    (train=32561, test=16281)
45222 if instances with unknown values are removed (train=30162, test=15060)
Duplicate or conflicting instances : 6
Class probabilities for adult.all file
Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
Extraction was done by Barry Becker from the 1994 adult database.  A set of
 reasonably clean records was extracted using the following conditions:
 ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
''')

# helper function for data frame str / summary
def rstr(df):
    return df.shape, df.apply(lambda x: [x.unique()])

# helper function for pickling files
def pickle_path(filename):
    return(pickle_dir + '\\' + filename)

# random seed for test_train_split
seed=123

var_names = ['age'
           , 'workclass'
           , 'lfnlwgt'
           , 'education'
           , 'educationnum'
           , 'maritalstatus'
           , 'occupation'
           , 'relationship'
           , 'race'
           , 'sex'
           , 'lcapitalgain'
           , 'lcapitalloss'
           , 'hoursperweek'
           , 'nativecountry'
           , 'income']

vars_types = ['continuous'
           , 'nominal'
           , 'continuous'
           , 'nominal'
           , 'continuous'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'continuous'
           , 'continuous'
           , 'continuous'
           , 'nominal'
           , 'nominal']

class_col = 'income'
features = [vn for vn in var_names if vn != class_col]

if True:

    url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    s=requests.get(url).content
    adult_train = pd.read_csv(io.StringIO(s.decode('utf-8')), names=var_names)

    url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    s=requests.get(url).content
    adult_test = pd.read_csv(io.StringIO(s.decode('utf-8')), names=var_names, skiprows=1)

    # combine the two datasets and split them later with standard code
    frames = [adult_train, adult_test]
    adult = pd.concat(frames)

    # some tidying required
    adult.income = adult.income.str.replace('.', '')
    for f, t in zip(var_names, vars_types):
        if t == 'continuous':
            adult[f] = adult[f].astype('int32')
        else:
            adult[f] = adult[f].str.replace(' ', '')
    qm_to_unk = lambda w: 'Unknown' if w == '?' else w
    tt_fix = lambda w: 'Trinidad and Tobago' if w == 'Trinadad&Tobago' else w
    adult['workclass'] = adult.workclass.apply(qm_to_unk)
    adult['nativecountry'] = adult.nativecountry.apply(qm_to_unk)
    adult['nativecountry'] = adult.nativecountry.apply(tt_fix)

    lending['lcaptialgain'] = np.log(lending['lcaptialgain'] + abs(lending['lcaptialgain'].min()) + 1)
    lending['lcaptialloss'] = np.log(lending['lcaptialloss'] + abs(lending['lcaptialloss'].min()) + 1)
    lending['lfnlwgt'] = np.log(lending['lfnlwgt'] + abs(lending['lfnlwgt'].min()) + 1)

    # create a small set that is easier to play with on a laptop
    adult_samp = adult.sample(frac=0.25, random_state=seed).reset_index()
    adult_samp.drop(labels='index', axis=1, inplace=True)

    adult.to_csv(pickle_path('adult.csv.gz'), index=False, compression='gzip')
    adult_samp.to_csv(pickle_path('adult_samp.csv.gz'), index=False, compression='gzip')
    

adult = pd.read_csv(pickle_path('adult.csv.gz'), compression='gzip')


# the following creates a copy of the data frame with int mappings of categorical variables for scikit-learn
# and also a dictionary containing the label encoders/decoders for each column
adult_pre = pd.DataFrame.copy(adult)

le_dict = {}
vars_dict = {}
onehot_features = []
onehot_dict = {}

for v, t in zip(var_names, vars_types):
    if t == 'nominal':
        # create a label encoder for all categoricals
        le_dict[v] = LabelEncoder().fit(adult[v].unique())
        # create a dictionary of categorical names
        names = list(le_dict[v].classes_)
        # transform each categorical column
        adult_pre[v] = le_dict[v].transform(adult[v])
        # create the reverse lookup
        for n in names:
            onehot_dict[v + '_' + str(n)] = v
    else:
        adult_pre[v] = adult[v]

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
X, y = adult_pre[features], adult_pre[class_col]

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

train_priors = y_train.value_counts().sort_index()/len(y_train)
test_priors = y_test.value_counts().sort_index()/len(y_test)

# one hot encoding required for classifier
# otherwise integer vectors will be treated as ordinal
# OneHotEncoder takes an integer list as an argument to state which columns to encode
encoder = OneHotEncoder(categorical_features=[i for i, (f, t) in enumerate(zip(var_names, vars_types)) if t == 'nominal' and vars_dict[f]['class_col'] == False])
encoder.fit(adult_pre.as_matrix())
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
import the adult.csv file
create the pandas dataframe and prints head: adult
create the categorical var encoder dictionary: le_dict
create a function to get any code for a column name and label: get_code
create the dictionary of categorical values: categories
create lists and dicts of one hot encoded variable names, onehot_features, onehot_dict
create the list of class names: class_names
create the pandas dataframe with encoded vars: adult_pre
create the pandas dataframe containing all features less class: X
create the pandas series containing the class 'decision': y
create the training and test sets: X_train, y_train, X_test, y_test
evaluate the training and test set priors and print them: train_priors, test_priors
create a One Hot Encoder and encode the train set: X_train_enc
(avoids treating variables as ordinal or continuous)
pickles objects that are needed by later steps: encoder, X_train_enc, y_train
creates a closure with the location of the pickle files for easy access to the stored datasets: pickle_path()
''')

print("adult.head()")
print(adult.head())

shp, variables = rstr(adult)
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
