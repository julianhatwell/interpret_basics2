import pandas as pd
import numpy as np
from datetime import datetime
import julian

spiel = '''
Data Set Information:
These files provide detailed road safety data about the circumstances of personal injury road accidents in GB from 1979, the types (including Make and Model) of vehicles involved and the consequential casualties. The statistics relate only to personal injury accidents on public roads that are reported to the police, and subsequently recorded, using the STATS19 accident reporting form.

All the data variables are coded rather than containing textual strings. The lookup tables are available in the "Additional resources" section towards the bottom of the table.

Please note that the 2015 data were revised on the 29th September 2016.

Accident, Vehicle and Casualty data for 2005 - 2009 are available in the time series files under 2014. Data for 1979 - 2004 are available as a single download under 2004 below.

Also includes: Results of breath-test screening data from recently introduced digital breath testing devices, as provided by Police Authorities in England and Wales

Results of blood alcohol levels (milligrams / 100 millilitres of blood) provided by matching coroners’ data (provided by Coroners in England and Wales and by Procurators Fiscal in Scotland) with fatality data from the STATS19 police data of road accidents in Great Britain. For cases when the Blood Alcohol Levels for a fatality are "unknown" are a consequence of an unsuccessful match between the two data sets.

Data clean up by James Brooke
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
           '1st_Road_Class' : np.uint8, '1st_Road_Number' : np.uint16, 'Road_Type' : np.uint8, 'Speed_limit' : np.float16,
           'Junction_Detail' : np.uint8, 'Junction_Control' : np.uint8, '2nd_Road_Class' : np.uint8,
           '2nd_Road_Number' : np.uint16, 'Pedestrian_Crossing-Human_Control' : np.uint8,
           'Pedestrian_Crossing-Physical_Facilities' : np.uint8, 'Light_Conditions' : np.uint8,
           'Weather_Conditions' : np.uint8, 'Road_Surface_Conditions' : np.uint8,
           'Special_Conditions_at_Site' : np.uint8, 'Carriageway_Hazards' : np.uint8,
           'Urban_or_Rural_Area' : np.uint8, 'Did_Police_Officer_Attend_Scene_of_Accident' : np.uint8,
           'LSOA_of_Accident_Location' : object}

    # recode class_col
    accident['Accident_Severity'] = accident['Accident_Severity'].replace({1 : 'Fatal', 2 : 'Serious', 3 : 'Slight'})

    # convert date and time to julian
    accident['Date_j'] = pd.Series([(str(d), str(t)) for d, t in zip(accident['Date'], accident['Time'])]).map(jul_conv)
    # tidy where necessary
    accident['Local_Authority_(Highway)'] = accident['Local_Authority_(Highway)'].str.slice(stop=3)
    # and drop unecessary/noisy columns
    accident.drop(labels=['Date', 'Time', 'Accident_Index', 'LSOA_of_Accident_Location'], axis=1, inplace=True)

    # get rid of na
    accident = accident.fillna(0.0)

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

    samp = accident.sample(frac=0.01, random_state=random_state).reset_index()
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
