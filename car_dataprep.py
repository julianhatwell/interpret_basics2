import urllib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
import pickle
from itertools import chain

pickle_dir = 'car_pickles'

print('''
M. Bohanec and V. Rajkovic: Knowledge acquisition and explanation for
multi-attribute decision making. In 8th Intl Workshop on Expert
Systems and their Applications, Avignon, France. pages 59-78, 1988.

Within machine-learning, this dataset was used for the evaluation
of HINT (Hierarchy INduction Tool), which was proved to be able to
completely reconstruct the original hierarchical model. This,
together with a comparison with C4.5, is presented in

B. Zupan, M. Bohanec, I. Bratko, J. Demsar: Machine learning by
function decomposition. ICML-97, Nashville, TN. 1997 (to appear)
''')

# helper function for data frame str / summary
def rstr(df):
    return df.shape, df.apply(lambda x: [x.unique()])

# helper function for pickling files
def pickle_path(filename):
    return(pickle_dir + '\\' + filename)

# random seed for test_train_split
seed=123

target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'

var_names = ['buying'
            , 'maint'
            , 'doors'
            , 'persons'
            , 'lug_boot'
            , 'safety'
            , 'acceptability']

vars_types = ['nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal'
           , 'nominal']

class_col = 'acceptability'
features = [vn for vn in var_names if vn != class_col]

car_bytes = urllib.request.urlopen(target_url)
car = pd.read_csv(car_bytes, header=None, names=var_names)
# recode to a 2 class subproblems
car.acceptability.loc[car.acceptability != 'unacc'] = 'acc'

# the following creates a copy of the data frame with int mappings of categorical variables for scikit-learn
# and also a dictionary containing the label encoders/decoders for each column
car_pre = pd.DataFrame.copy(car)

le_dict = {}
vars_dict = {}
onehot_features = []
onehot_dict = {}

for i, v in enumerate(var_names):
    # create a label encoder for all categoricals
    le_dict[v] = LabelEncoder().fit(car[v].unique())

    # transform each categorical column
    car_pre[v] = le_dict[v].transform(car[v])

    # create a dictionary of categorical names
    names = list(le_dict[v].classes_)

    vars_dict[v] = {'labels' : names
                    , 'onehot_labels' : [v + '_' + n for n in names]
                    , 'class_col' : True if v == class_col else False
                    , 'data_type' : vars_types[i]}

    for n in names:
        onehot_dict[v + '_' + n] = v

categorical_features=[i for i, (c, t) in enumerate(zip([vars_dict[f]['class_col'] for f in features],
[vars_dict[f]['data_type'] == 'nominal' for f in features])) if not c and t]

# creates a flat list just for the features
onehot_features = list(chain.from_iterable([vars_dict[f]['onehot_labels'] for f in features]))

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
X, y = car_pre[features], car_pre[class_col]

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

train_priors = y_train.value_counts().sort_index()/len(y_train)
test_priors = y_test.value_counts().sort_index()/len(y_test)

# one hot encoding required for classifier
# otherwise integer vectors will be treated as ordinal
# OneHotEncoder takes an integer list as an argument to state which columns to encode
encoder = OneHotEncoder(categorical_features=categorical_features)
encoder.fit(car_pre.as_matrix())
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
import the car.csv file
create the pandas dataframe and prints head: car
create the categorical var encoder dictionary: le_dict
create a function to get any code for a column name and label: get_code
create the dictionary of categorical values: categories
creates the list of one hot encoded variable names, onehot_features
create the list of class names: class_names
create the pandas dataframe with encoded vars: car_pre
create the pandas dataframe containing all features less class: X
create the pandas series containing the class 'decision': y
create the training and test sets: X_train, y_train, X_test, y_test
evaluate the training and test set priors and print them: train_priors, test_priors
create a One Hot Encoder and encode the train set: X_train_enc
(avoids treating variables as ordinal or continuous)
pickles objects that are needed by later steps: encoder, X_train_enc, y_train
creates a closure with the location of the pickle files for easy access to the stored datasets: pickle_path()
''')

print("car.head()")
print(car.head())

shp, variables = rstr(car)
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
