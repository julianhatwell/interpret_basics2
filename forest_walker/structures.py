import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from itertools import chain

if sys.platform == 'win32':
    path_sep = '\\'
else:
    path_sep = '/'


class data_container:

    def __init__(self
    , data
    , class_col
    , var_names
    , var_types = None
    , pickle_dir = ''
    , random_state = 123
    , spiel = ''):
        self.spiel = spiel
        self.random_state = random_state

        self.data = data
        self.data_pre = pd.DataFrame.copy(self.data)
        self.class_col = class_col
        self.var_names = var_names
        self.var_types = var_types
        self.pickle_dir = pickle_dir

        self.features = [vn for vn in var_names if vn != class_col]

        self.le_dict = {}
        self.vars_dict = {}
        self.onehot_dict = {}

        for v, t in zip(self.var_names, self.var_types):
            if t == 'nominal':
                # create a label encoder for all categoricals
                self.le_dict[v] = LabelEncoder().fit(self.data[v].unique())
                # create a dictionary of categorical names
                names = list(self.le_dict[v].classes_)
                # transform each categorical column
                self.data_pre[v] = self.le_dict[v].transform(self.data[v])
                # create the reverse lookup
                for n in names:
                    self.onehot_dict[v + '_' + str(n)] = v
            else:
                self.data_pre[v] = self.data[v]

            self.vars_dict[v] = {'labels' : names if t == 'nominal' else None
                                , 'onehot_labels' : [v + '_' + str(n) for n in names] if t == 'nominal' else None
                                , 'class_col' : True if v == class_col else False
                                , 'data_type' : t}
        del names
        del t

        self.categorical_features=[i for i, (c, t) in enumerate(zip([self.vars_dict[f]['class_col'] for f in self.features],
        [self.vars_dict[f]['data_type'] == 'nominal' for f in self.features])) if not c and t]

        # creates a flat list just for the features
        self.onehot_features = []
        self.continuous_features = []
        for f, t in zip(self.var_names, self.var_types):
            if f == self.class_col: continue
            if t == 'continuous':
                self.continuous_features.append(f)
            else:
                self.onehot_features.append(self.vars_dict[f]['onehot_labels'])

        # They get stuck on the end by encoding
        self.onehot_features.append(self.continuous_features)
        # flatten out the nesting
        self.onehot_features = list(chain.from_iterable(self.onehot_features))

        self.class_names = list(self.le_dict[self.class_col].classes_)

    # helper function for pickling files
    def pickle_path(self, filename):
        return(self.pickle_dir + path_sep + filename)

    # helper function for data frame str / summary
    def rstr(self):
        return(self.data.shape, self.data.apply(lambda x: [x.unique()]))

    # a function to return any code from a label
    def get_code(self, col, label):
        return self.le_dict[col].transform([label])[0]

    # a function to return any label from a code
    def get_label(self, col, label):
        return self.le_dict[col].inverse_transform([label])[0]

    # train test splitting
    def tt_split(self, test_size=0.3, random_state=None):
        if random_state is not None:
            random_state = self.random_state
        X, y = self.data_pre[self.features], self.data_pre[self.class_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        train_priors = y_train.value_counts().sort_index()/len(y_train)
        test_priors = y_test.value_counts().sort_index()/len(y_test)

        # one hot encoding required for classifier
        # otherwise integer vectors will be treated as ordinal
        # OneHotEncoder takes an integer list as an argument to state which columns to encode
        encoder = OneHotEncoder(categorical_features=self.categorical_features)
        encoder.fit(self.data_pre.as_matrix())
        X_train_enc = encoder.transform(X_train)

        return({
        'X_train': X_train,
        'X_train_enc' : X_train_enc,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'encoder' : encoder})
