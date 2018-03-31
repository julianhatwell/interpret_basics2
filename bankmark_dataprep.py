import pandas as pd
import numpy as np
from datetime import datetime
import julian

spiel = '''
Data Set Information:
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

There are four datasets:
1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).


Attribute Information:

Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
'''

if True:
    '''
    jul_conv = lambda x : 0 if x[0] == 'nan' or x[1] == 'nan' else julian.to_jd(datetime.strptime(x[0] + ' ' + x[1], '%d/%m/%Y %M:%S'))

    # random seed for train test split and sampling
    random_state = 123

    vtypes = {'Accident_Index' : object, 'Location_Easting_OSGR' : np.float64, 'Location_Northing_OSGR' : np.float64,
           'Longitude' : np.float64, 'Latitude' : np.float64, 'Police_Force' : np.uint8, 'Accident_Severity' : np.uint8,
           'Number_of_Vehicles' : np.uint8, 'Number_of_Casualties' : np.uint8, 'Date' : object, 'Day_of_Week' : np.uint8,
           'Time' : object, 'Local_Authority_(District)' : np.uint16, 'Local_Authority_(Highway)' : object,
           '1st_Road_Class' : np.uint8, '1st_Road_Number' : np.uint16, 'Road_Type' : np.float16, 'Speed_limit' : np.float16,
           'Junction_Detail' : np.float16, 'Junction_Control' : np.float16, '2nd_Road_Class' : np.float16,
           '2nd_Road_Number' : np.float16, 'Pedestrian_Crossing-Human_Control' : np.float16,
           'Pedestrian_Crossing-Physical_Facilities' : np.float16, 'Light_Conditions' : np.float16,
           'Weather_Conditions' : np.float16, 'Road_Surface_Conditions' : np.float16,
           'Special_Conditions_at_Site' : np.float16, 'Carriageway_Hazards' : np.float16,
           'Urban_or_Rural_Area' : np.uint8, 'Did_Police_Officer_Attend_Scene_of_Accident' : np.uint8,
           'LSOA_of_Accident_Location' : object}

    accident = pd.read_csv('data_source_files\\Accidents.csv', dtype=vtypes, na_values=-1, low_memory=False)

    # convert date and time to julian
    accident['Date_j'] = pd.Series([(str(d), str(t)) for d, t in zip(accident['Date'], accident['Time'])]).map(jul_conv)
    accident.drop(labels=['Date', 'Time', 'Accident_Index'], axis=1, inplace=True)

    # rearrange so class col at end
    class_col = 'Accident_Severity'
    pos = np.where(accident.columns == class_col)[0][0]
    var_names = list(accident.columns[:pos]) + list(accident.columns[pos + 1:]) + list(accident.columns[pos:pos + 1])
    accident = accident[var_names]

    # save
    accident.to_csv('forest_surveyor\\datafiles\\accident.csv.gz', index=False, compression='gzip')


    # create small set that is easier to play with on a laptop
    samp = accident.sample(frac=0.1, random_state=random_state).reset_index()
    samp.drop(labels='index', axis=1, inplace=True)
    samp.to_csv('forest_surveyor\\datafiles\\accident_samp.csv.gz', index=False, compression='gzip')

    samp = mydata.data.sample(frac=0.01, random_state=random_state).reset_index()
    samp.drop(labels='index', axis=1, inplace=True)
    samp.to_csv('forest_surveyor\\datafiles\\accident_small_samp.csv.gz', index=False, compression='gzip')
    '''

accident = pd.read_csv(pickle_path('lend_samp.csv.gz'), compression='gzip')
var_names = accident.columns
vars_types = ['nominal' if dt.name == 'object' else 'continuous' for dt in accident.dtypes.values]
features = [vn for vn in var_names if vn != class_col]

# the following creates a copy of the data frame with int mappings of categorical variables for scikit-learn
# and also a dictionary containing the label encoders/decoders for each column
accident_pre = pd.DataFrame.copy(accident)

le_dict = {}
vars_dict = {}
onehot_dict = {}

for v, t in zip(var_names, vars_types):
    if t == 'nominal':
        # create a label encoder for all categoricals
        le_dict[v] = LabelEncoder().fit(accident[v].unique())
        # create a dictionary of categorical names
        names = list(le_dict[v].classes_)
        # transform each categorical column
        accident_pre[v] = le_dict[v].transform(accident[v])
        # create the reverse lookup
        for n in names:
            onehot_dict[v + '_' + str(n)] = v
    else:
        accident_pre[v] = accident[v]

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
X, y = accident_pre[features], accident_pre[class_col]

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

train_priors = y_train.value_counts().sort_index()/len(y_train)
test_priors = y_test.value_counts().sort_index()/len(y_test)

# one hot encoding required for classifier
# otherwise integer vectors will be treated as ordinal
# OneHotEncoder takes an integer list as an argument to state which columns to encode
encoder = OneHotEncoder(categorical_features=categorical_features)
encoder.fit(accident_pre.as_matrix())
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
import the accident.csv file
create the pandas dataframe and prints head: accident
create the categorical var encoder dictionary: le_dict
create a function to get any code for a column name and label: get_code
create the dictionary of categorical values: categories
creates the list of one hot encoded variable names, onehot_features
create the list of class names: class_names
create the pandas dataframe with encoded vars: accident_pre
create the pandas dataframe containing all features less class: X
create the pandas series containing the class 'decision': y
create the training and test sets: X_train, y_train, X_test, y_test
evaluate the training and test set priors and print them: train_priors, test_priors
create a One Hot Encoder and encode the train set: X_train_enc
(avoids treating variables as ordinal or continuous)
pickles objects that are needed by later steps: encoder, X_train_enc, y_train
creates a closure with the location of the pickle files for easy access to the stored datasets: pickle_path()
''')

print("accident.head()")
print(accident.head())

shp, variables = rstr(accident)
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
