#get rid
import time
import timeit
#ok
import sys
import math
import multiprocessing as mp
import numpy as np
from forest_surveyor import p_count, p_count_corrected
from pandas import DataFrame, Series
from pyfpgrowth import find_frequent_patterns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import deque, defaultdict
from scipy import sparse
from scipy.stats import sem, entropy
from operator import itemgetter
from itertools import chain
from copy import deepcopy
from scipy.stats import chi2_contingency
from forest_surveyor import config as cfg
from forest_surveyor.async_structures import as_tree_walk

class default_encoder:

    def transform(x):
        return(sparse.csr_matrix(x))
    def fit(x):
        return(x)

class data_container:

    def __init__(self
    , data
    , class_col
    , var_names = None
    , var_types = None
    , project_dir = None
    , pickle_dir = ''
    , random_state = None
    , spiel = ''):
        self.spiel = spiel
        if random_state is None:
            self.random_state = 123
        else:
            self.random_state = random_state

        self.data = data
        self.data_pre = DataFrame.copy(self.data)
        self.class_col = class_col
        self.pickle_dir = pickle_dir

        if project_dir is None:
            self.project_dir = cfg.project_dir
        else:
            self.project_dir = project_dir

        if var_names is None:
            self.var_names = list(self.data.columns)
        else:
            self.var_names = var_names

        if var_types is None:
            self.var_types = ['nominal' if dt.name == 'object' else 'continuous' for dt in self.data.dtypes.values]
        else:
            self.var_types = var_types

        self.features = [vn for vn in self.var_names if vn != self.class_col]
        self.class_names = list(self.data[self.class_col].unique())

        self.le_dict = {}
        self.var_dict = {}
        self.onehot_dict = {}

        for i, (v, t) in enumerate(zip(self.var_names, self.var_types)):
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

            self.var_dict[v] = {'labels' : names if t == 'nominal' else None
                                , 'onehot_labels' : [v + '_' + str(n) for n in names] if t == 'nominal' else None
                                , 'class_col' : True if v == class_col else False
                                , 'data_type' : t
                                , 'order_col' : i}

        if any(n == 'nominal' for n in self.var_types ): del names
        del t

        self.categorical_features=[i for i, (c, t) in enumerate(zip([self.var_dict[f]['class_col'] for f in self.features],
        [self.var_dict[f]['data_type'] == 'nominal' for f in self.features])) if not c and t]

        # creates a flat list just for the features
        self.onehot_features = []
        self.continuous_features = []
        for f, t in zip(self.var_names, self.var_types):
            if f == self.class_col: continue
            if t == 'continuous':
                self.continuous_features.append(f)
            else:
                self.onehot_features.append(self.var_dict[f]['onehot_labels'])

        # They get stuck on the end by encoding
        self.onehot_features.append(self.continuous_features)
        # flatten out the nesting
        self.onehot_features = list(chain.from_iterable(self.onehot_features))

        # one hot encoding required for classifier
        # otherwise integer vectors will be treated as ordinal
        # OneHotEncoder takes an integer list as an argument to state which columns to encode
        # If no nominal vars, then simply convert to sparse matrix format
        if len(self.categorical_features) > 0:
            encoder = OneHotEncoder(categorical_features=self.categorical_features)
            encoder.fit(self.data_pre.as_matrix())
            self.encoder = encoder
        else:
            self.encoder = default_encoder

    # helper function for pickling files
    def pickle_path(self, filename = ''):
        if len(self.project_dir) > 0:
            return(self.project_dir + cfg.path_sep + self.pickle_dir + cfg.path_sep + filename)
        else:
            return(self.pickle_dir + cfg.path_sep + filename)

    # helper function for data frame str / summary
    def rstr(self):
        return(self.data.shape, self.data.apply(lambda x: [x.unique()]))

    # a function to return any code from a label
    def get_code(self, col, label):
        if len(self.le_dict.keys()) > 0 and label in self.le_dict.keys():
            return self.le_dict[col].transform([label])[0]
        else:
            return(label)

    # a function to return any label from a code
    def get_label(self, col, label):
        if len(self.le_dict.keys()) > 0 and col in self.le_dict.keys():
            return self.le_dict[col].inverse_transform([label])[0]
        else:
            return(label)

    # train test splitting
    def tt_split(self, test_size=0.3, random_state=None):
        if random_state is None:
            random_state = self.random_state
        X, y = self.data_pre[self.features], self.data_pre[self.class_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        train_priors = y_train.value_counts().sort_index()/len(y_train)
        test_priors = y_test.value_counts().sort_index()/len(y_test)

        X_train_enc = self.encoder.transform(X_train)

        return({
        'X_train': X_train,
        'X_train_enc' : X_train_enc,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'encoder' : self.encoder,
        'train_priors' : train_priors,
        'test_priors' : test_priors})

    def xval_split(self, iv_low, iv_high, test_index, random_state=None):
        iv_high = int(iv_high)
        iv_low = int(iv_low)
        test_index = int(test_index)

        if test_index >= iv_high or test_index < iv_low:
            print('test_index out of range. Setting to lowest value in range (iv_low)')
            test_index = iv_low

        if random_state is None:
            random_state = self.random_state

        # generate an indexing vector
        iv = np.random.RandomState(random_state).randint(low=iv_low, high=iv_high, size=np.shape(self.data_pre)[0])

        # data in readiness
        X, y = self.data_pre[self.features], self.data_pre[self.class_col]

        # perform the split
        X_test = X.iloc[iv == test_index]
        y_test = y.iloc[iv == test_index]
        X_train = X.iloc[iv != test_index]
        y_train = y.iloc[iv != test_index]

        train_priors = y_train.value_counts().sort_index()/len(y_train)
        test_priors = y_test.value_counts().sort_index()/len(y_test)

        X_train_enc = self.encoder.transform(X_train)

        return({
        'X_train': X_train,
        'X_train_enc' : X_train_enc,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'encoder' : self.encoder,
        'train_priors' : train_priors,
        'test_priors' : test_priors})

    def pretty_rule(self, rule):
        Tr_Fa = lambda x, y, z : x + ' True' if ~y else x + ' False'
        lt_gt = lambda x, y, z : x + ' <= ' + str(z) if y else x + ' > ' + str(z)
        def bin_or_cont(x, y, z, onehot_dict):
            if x in onehot_dict:
                return(Tr_Fa(x,y,z))
            else:
                return(lt_gt(x,y,z))
        return(' AND '.join([bin_or_cont(f, t, v, self.onehot_dict) for f, t, v in rule]))

class instance_paths_container:
    def __init__(self
    , paths
    , tree_preds
    , patterns=None
    , instance_id=None):
        self.paths = paths
        self.tree_preds = tree_preds
        self.patterns = patterns
        self.instance_id = instance_id

    def discretize_paths(self, var_dict, bins=4, equal_counts=False):
        # check if bins is not numeric or can't be cast, then force equal width (equal_counts = False)
        if equal_counts:
            def hist_func(x, bins):
                npt = len(x)
                return np.interp(np.linspace(0, npt, bins + 1),
                                 np.arange(npt),
                                 np.sort(x))
        else:
            def hist_func(x, bins):
                return(np.histogram(x, bins))

        cont_vars = [vn for vn in var_dict if var_dict[vn]['data_type'] == 'continuous' and var_dict[vn]['class_col'] == False]
        for feature in cont_vars:
        # nan warnings OK, it just means the less than or greater than test was never used
            # lower bound, greater than
            lowers = [item[2] for nodes in self.paths for item in nodes if item[0] == feature and item[1] == False]

            # upper bound, less than
            uppers = [item[2] for nodes in self.paths for item in nodes if item[0] == feature and item[1] == True]

            upper_bins = np.histogram(uppers, bins=bins)[1]
            lower_bins = np.histogram(lowers, bins=bins)[1]

            # upper_bin_midpoints = pd.Series(upper_bins).rolling(window=2, center=False).mean().values[1:]
            upper_bin_means = (np.histogram(uppers, upper_bins, weights=uppers)[0] /
                                np.histogram(uppers, upper_bins)[0]).round(5)

            # lower_bin_midpoints = pd.Series(lower_bins).rolling(window=2, center=False).mean().values[1:]
            lower_bin_means = (np.histogram(lowers, lower_bins, weights=lowers)[0] /
                                np.histogram(lowers, lower_bins)[0]).round(5)

            # discretize functions from histogram means
            upper_discretize = lambda x: upper_bin_means[np.max([np.min([np.digitize(x, upper_bins), len(upper_bin_means)]), 1]) - 1]
            lower_discretize = lambda x: lower_bin_means[np.max([np.min([np.digitize(x, lower_bins, right= True), len(upper_bin_means)]), 1]) - 1]

            paths_discretized = []
            for nodes in self.paths:
                nodes_discretized = []
                for f, t, v in nodes:
                    if f == feature:
                        if t == False: # greater than, lower bound
                            v = lower_discretize(v)
                        else:
                            v = upper_discretize(v)
                    nodes_discretized.append((f, t, v))
                paths_discretized.append(nodes_discretized)
            # at the end of each loop, update the instance variable
            self.paths = paths_discretized

    def mine_patterns(self, support=0.1):
        # convert to an absolute number of instances rather than a fraction
        if support < 1:
            support = round(support * len(self.paths))
        self.patterns = find_frequent_patterns(self.paths, support)

    def sort_patterns(self, alpha=0.0, weights=None):
        alpha = float(alpha)
        if weights is None:
            weights = [1] * len(self.patterns)
        fp_scope = self.patterns.copy()
        # to shrink the support of shorter freq_patterns
        # formula is sqrt(weight) * log(sup * (len - alpha) / len)
        score_function = lambda x, w: (x[0], x[1], math.sqrt(w) * math.log(x[1]) * (len(x[0]) - alpha) / len(x[0]))
        fp_scope = [fp for fp in map(score_function, fp_scope.items(), weights)]
        # score is now at position 2 of tuple
        self.patterns = sorted(fp_scope, key=itemgetter(2), reverse = True)

class batch_paths_container:
    def __init__(self
    , path_detail
    , by_tree):
        self.by_tree = by_tree
        self.path_detail = path_detail

    def flip(self):
        n = len(self.path_detail[0])
        flipped_paths = [[]] * n
        for i in range(n):
            flipped_paths[i] =  [p[i] for p in self.path_detail]
        self.path_detail = flipped_paths
        self.by_tree = not self.by_tree

    def major_class_from_paths(self, batch_idx, return_counts=True):
        if self.by_tree:
            pred_classes = [self.path_detail[p][batch_idx]['pred_class'] for p in range(len(self.path_detail))]
        else:
            pred_classes = [self.path_detail[batch_idx][p]['pred_class'] for p in range(len(self.path_detail[batch_idx]))]

        unique, counts = np.unique(pred_classes, return_counts=True)

        if return_counts:
            return(unique[np.argmax(counts)], dict(zip(unique, counts)))
        else: return(unique[np.argmax(counts)])

    def get_instance_paths(self, batch_idx, which_trees='all', feature_values=True):
        # It is a sequence number in the batch of instance_paths that have been walked
        # Each path is the route of one instance down one tree

        batch_idx = math.floor(batch_idx) # make sure it's an integer
        true_to_lt = lambda x: '<' if x == True else '>'

        # extract the paths we want by filtering on tree performance
        if self.by_tree:
            n_paths = len(self.path_detail)
            if which_trees == 'correct':
                paths_info = [self.path_detail[pd][batch_idx]['path'] for pd in range(n_paths) if self.path_detail[pd][batch_idx]['tree_correct']]
            elif which_trees == 'majority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info = [self.path_detail[pd][batch_idx]['path'] for pd in range(n_paths) if self.path_detail[pd][batch_idx]['pred_class'] == major_class]
            elif which_trees == 'minority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info = [self.path_detail[pd][batch_idx]['path'] for pd in range(n_paths) if self.path_detail[pd][batch_idx]['pred_class'] != major_class]
            else:
                paths_info = [self.path_detail[pd][batch_idx]['path'] for pd in range(n_paths)]
        else:
            n_paths = len(self.path_detail[batch_idx])
            if which_trees == 'correct':
                paths_info = [self.path_detail[batch_idx][pd]['path'] for pd in range(n_paths) if self.path_detail[batch_idx][pd]['tree_correct']]
            elif which_trees == 'majority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info = [self.path_detail[batch_idx][pd]['path'] for pd in range(n_paths) if self.path_detail[batch_idx][pd]['pred_class'] == major_class]
            elif which_trees == 'minority':
                major_class = self.major_class_from_paths(batch_idx, return_counts=False)
                paths_info = [self.path_detail[batch_idx][pd]['path'] for pd in range(n_paths) if self.path_detail[batch_idx][pd]['pred_class'] != major_class]
            else:
                paths_info = [self.path_detail[batch_idx][pd]['path'] for pd in range(n_paths)]

        # path formatting - should it be on values level or features level
        if feature_values:
            paths = [[]] * len(paths_info)
            for i, p in enumerate(paths_info):
                paths[i] = [(f, leq, t) for f, leq, t in zip(p['feature_name'], p['leq_threshold'], p['threshold'])]
        else:
            paths = [p['feature_name'] for p in paths_info]

        # tree performance stats
        if self.by_tree:
            tree_preds = [self.path_detail[t][batch_idx]['pred_class_label'] for t in range(n_paths)]
        else:
            tree_preds = [self.path_detail[batch_idx][t]['pred_class_label'] for t in range(n_paths)]

        # return an object for requested instance
        instance_paths = instance_paths_container(paths, tree_preds)
        return(instance_paths)

class forest_walker:

    def __init__(self
    , forest
    , data_container
    , encoder = None
    , prediction_model = None):
        self.forest = forest
        self.features = data_container.onehot_features
        self.n_features = len(self.features)
        if data_container.class_col in data_container.le_dict.keys():
            self.class_names = data_container.get_label(data_container.class_col, [i for i in range(len(data_container.class_names))])
            self.get_label = data_container.get_label
            self.class_col = data_container.class_col
        else:
            self.class_names = data_container.class_names
            self.get_label = None
        self.encoder = encoder
        if prediction_model is None:
            self.prediction_model = forest
        else: self.prediction_model = prediction_model

        # base counts for all trees
        self.root_features = np.zeros(len(self.features)) # set up a 1d feature array to count features appearing as root nodes
        self.child_features = np.zeros(len(self.features))
        self.lower_features = np.zeros(len(self.features))
        self.structure = {'root_features' : self.root_features
                         , 'child_features' : self.child_features
                         , 'lower_features' : self.lower_features}

        # walk through each tree to get the structure
        for t, tree in enumerate(self.forest.estimators_):

            # root, child and lower counting, one time only (first class)
            structure = tree.tree_
            feature = structure.feature
            children_left = structure.children_left
            children_right = structure.children_right

            self.root_features[feature[0]] += 1
            if children_left[0] >= 0:
                self.child_features[feature[children_left[0]]] +=1
            if children_right[0] >= 0:
                self.child_features[feature[children_right[0]]] +=1

            for j, f in enumerate(feature):
                if j < 3: continue # root and children
                if f < 0: continue # leaf nodes
                self.lower_features[f] += 1

    def full_survey(self
        , instances
        , labels):

        self.instances = instances
        self.labels = labels
        self.n_instances = instances.shape[0]
        self.n_classes = len(np.unique(labels))

        if labels is not None:
            if len(labels) != self.n_instances:
                raise ValueError("labels and instances must be same length")

        trees = self.forest.estimators_
        self.n_trees = len(trees)

        self.feature_depth = np.full((self.n_instances, self.n_trees, self.n_features), np.nan) # set up a 1d feature array for counting
        self.tree_predictions = np.full((self.n_instances, self.n_trees), np.nan)
        self.tree_performance = np.full((self.n_instances, self.n_trees), np.nan)
        self.path_lengths = np.zeros((self.n_instances, self.n_trees))

        # walk through each tree
        for t, tree in enumerate(trees):
            # get the feature vector out of the tree object
            feature = tree.tree_.feature

            self.tree_predictions[:, t] = tree.predict(self.instances)
            self.tree_performance[:, t] = self.tree_predictions[:, t] == self.labels

            # extract path and get path lengths
            path = tree.decision_path(self.instances).indices
            paths_begin = np.where(path == 0)
            paths_end = np.append(np.where(path == 0)[0][1:], len(path))
            self.path_lengths[:, t] = paths_end - paths_begin

            depth = 0
            instance = -1
            for p in path:
                if feature[p] < 0: # leaf node
                    # TO DO: what's in a leaf node
                    continue
                if p == 0: # root node
                    instance += 1 # a new instance
                    depth = 0 # a new path
                else:
                    depth += 1 # same instance, descends tree one more node
                self.feature_depth[instance][t][feature[p]] = depth

    def forest_stats_by_label(self, label = None):
        if label is None:
            idx = Series([True] * self.n_instances) # it's easier if has the same type as the labels
            label = 'all_classes'
        else:
            idx = self.labels == label
        idx = idx.values

        n_instances_lab = sum(idx) # number of instances having the current label
        if n_instances_lab == 0: return

        # object to hold all the statistics
        statistics = {}
        statistics['n_trees'] = self.n_trees
        statistics['n_instances'] = n_instances_lab

        # get a copy of the arrays, containing only the required instances
        feature_depth_lab = self.feature_depth[idx]
        path_lengths_lab = self.path_lengths[idx]
        tree_performance_lab = self.tree_performance[idx]

        # gather statistics from the feature_depth array, for each class label
        # shape is instances, trees, features, so [:,:,fd]
        depth_counts = [np.unique(feature_depth_lab[:,:,fd][~np.isnan(feature_depth_lab[:,:,fd])], return_counts = True) for fd in range(self.n_features)]

        # number of times each feature node was visited
        statistics['n_node_traversals'] = np.array([np.nansum(dcz[1]) for dcz in depth_counts], dtype=np.float32)
        # number of times feature was a root node (depth == 0)
        statistics['n_root_traversals'] = np.array([depth_counts[dc][1][np.where(depth_counts[dc][0] == 0)][0] if depth_counts[dc][1][np.where(depth_counts[dc][0] == 0)] else 0 for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was a root-child (depth == 1)
        statistics['n_child_traversals'] = np.array([depth_counts[dc][1][np.where(depth_counts[dc][0] == 1)][0] if depth_counts[dc][1][np.where(depth_counts[dc][0] == 1)] else 0 for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was a lower node (depth > 1)
        statistics['n_lower_traversals'] = np.array([np.nansum(depth_counts[dc][1][np.where(depth_counts[dc][0] > 1)] if any(depth_counts[dc][1][np.where(depth_counts[dc][0] > 1)]) else 0) for dc in range(len(depth_counts))], dtype=np.float32)
        # number of times feature was not a root
        statistics['n_nonroot_traversals'] = statistics['n_node_traversals'] - statistics['n_root_traversals'] # total feature visits - number of times feature was a root

        # number of correct predictions
        statistics['n_correct_preds'] = np.sum(tree_performance_lab) # total number of correct predictions
        statistics['n_path_length'] = np.sum(path_lengths_lab) # total path length accumulated by each feature

        # above measures normalised over all features
        p_ = lambda x : x / np.nansum(x)

        statistics['p_node_traversals'] = p_(statistics['n_node_traversals'])
        statistics['p_root_traversals'] = p_(statistics['n_root_traversals'])
        statistics['p_nonroot_traversals'] = p_(statistics['n_nonroot_traversals'])
        statistics['p_child_traversals'] = p_(statistics['n_child_traversals'])
        statistics['p_lower_traversals'] = p_(statistics['n_lower_traversals'])
        statistics['p_correct_preds'] = np.mean(tree_performance_lab) # accuracy

        statistics['m_node_traversals'] = np.mean(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0) # mean number of times feature appeared over all instances
        statistics['m_root_traversals'] = np.mean(np.sum(feature_depth_lab == 0, axis = 1), axis = 0) # mean number of times feature appeared as a root node, over all instances
        statistics['m_nonroot_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0)
        statistics['m_child_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0)
        statistics['m_lower_traversals'] = np.mean(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0)
        statistics['m_feature_depth'] = np.mean(np.nanmean(feature_depth_lab, axis = 1), axis = 0) # mean depth of each feature when it appears
        statistics['m_path_length'] = np.mean(np.nanmean(path_lengths_lab, axis = 1), axis = 0) # mean path length of each instance in the forest
        statistics['m_correct_preds'] = np.mean(np.mean(tree_performance_lab, axis = 1)) # mean prop. of trees voting correctly per instance

        if n_instances_lab > 1: # can't compute these on just one example
            statistics['sd_node_traversals'] = np.std(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0, ddof = 1) # sd of number of times... over all instances and trees
            statistics['sd_root_traversals'] = np.std(np.sum(feature_depth_lab == 0, axis = 1), axis = 0, ddof = 1) # sd of number of times feature appeared as a root node, over all instances
            statistics['sd_nonroot_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0, ddof = 1) # sd of number of times feature appeared as a nonroot node, over all instances
            statistics['sd_child_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0, ddof = 1)
            statistics['sd_lower_traversals'] = np.std(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0, ddof = 1)
            statistics['sd_feature_depth'] = np.std(np.nanmean(feature_depth_lab, axis = 1), axis = 0, ddof = 1) # sd depth of each feature when it appears
            statistics['sd_path_length'] = np.std(np.nanmean(path_lengths_lab, axis = 1), axis = 0, ddof = 1)
            statistics['sd_correct_preds'] = np.std(np.mean(tree_performance_lab, axis = 1), ddof = 1) # std prop. of trees voting correctly per instance
            statistics['se_node_traversals'] = sem(np.sum(~np.isnan(feature_depth_lab), axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se of mean number of times feature appeared over all instances
            statistics['se_root_traversals'] = sem(np.sum(feature_depth_lab == 0, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se of mean of number of times feature appeared as a root node, over all instances
            statistics['se_nonroot_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) > 0, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # sd of number of times feature appeared as a nonroot node, over all instances
            statistics['se_child_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) == 1, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_lower_traversals'] = sem(np.sum(np.nan_to_num(feature_depth_lab) > 1, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_feature_depth'] = sem(np.nanmean(feature_depth_lab, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit') # se depth of each feature when it appears
            statistics['se_path_length'] = sem(np.nanmean(path_lengths_lab, axis = 1), axis = 0, ddof = 1, nan_policy = 'omit')
            statistics['se_correct_preds'] = sem(np.mean(tree_performance_lab, axis = 1), ddof = 1, nan_policy = 'omit') # se prop. of trees voting correctly per instance
        else:
            statistics['sd_node_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_root_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_nonroot_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_child_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_lower_traversals'] = np.full(self.n_features, np.nan)
            statistics['sd_feature_depth'] = np.full(self.n_features, np.nan)
            statistics['sd_path_length'] = np.full(self.n_features, np.nan)
            statistics['sd_correct_preds'] = np.full(self.n_features, np.nan)
            statistics['se_node_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_root_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_nonroot_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_child_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_lower_traversals'] = np.full(self.n_features, np.nan)
            statistics['se_feature_depth'] = np.full(self.n_features, np.nan)
            statistics['se_path_length'] = np.full(self.n_features, np.nan)
            statistics['se_correct_preds'] = np.full(self.n_features, np.nan)
        return(statistics)

    def forest_stats(self, class_labels = None):

        statistics = {}

        if class_labels is None:
            class_labels = np.unique(self.labels)
        for cl in class_labels:
            statistics[cl] = self.forest_stats_by_label(cl)

        statistics['all_classes'] = self.forest_stats_by_label()
        return(statistics)

    def tree_structures(self, tree, instances, labels, n_instances):

        # structural objects from tree
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        path = tree.decision_path(instances).indices

        # predictions from tree
        tree_pred = tree.predict(instances)
        tree_pred_proba = tree.predict_proba(instances)

        if labels is None:
            tree_correct = [None] * n_instances
        else:
            tree_correct = tree_pred == labels.values

        if labels is not None:
            tree_pred_labels = self.get_label(self.class_col, tree_pred.astype(int))
        else:
            tree_pred_labels = tree_pred

        return(tree_pred, tree_pred_labels, tree_pred_proba, tree_correct, feature, threshold, path)

    def forest_walk(self, instances, labels = None, async=False):

        features = self.features
        n_instances = instances.shape[0]
        instance_ids = instances.index.tolist()

        # encode features prior to sending into tree for path analysis
        if self.encoder is None:
            instances = np.matrix(instances)
            n_features = instances.shape[1]
        else:
            instances = self.encoder.transform(instances)
            if 'todense' in dir(instances): # it's a sparse matrix
                instances = instances.todense()
            n_features = instances.shape[1]

        if async:
            async_out = []
            n_cores = mp.cpu_count()-1
            pool = mp.Pool(processes=n_cores)

            for i, t in enumerate(self.forest.estimators_):

                # process the tree
                tree_pred, tree_pred_labels, \
                tree_pred_proba, tree_correct, \
                feature, threshold, path = self.tree_structures(t, instances, labels, n_instances)
                # walk the tree
                async_out.append(pool.apply_async(as_tree_walk,
                                                (i, instances, labels,
                                                instance_ids, n_instances,
                                                tree_pred, tree_pred_labels,
                                                tree_pred_proba, tree_correct,
                                                feature, threshold, path, features)
                                                ))

            # block and collect the pool
            pool.close()
            pool.join()

            # get the async results and sort to ensure original tree order and remove tree index
            tp = [async_out[j].get() for j in range(len(async_out))]
            tp.sort()
            tree_paths = [tp[k][1] for k in range(len(tp))]

        else:
            tree_paths = [[]] * len(self.forest.estimators_)
            for i, t in enumerate(self.forest.estimators_):

                # process the tree
                tree_pred, tree_pred_labels, \
                tree_pred_proba, tree_correct, \
                feature, threshold, path = self.tree_structures(t, instances, labels, n_instances)
                # walk the tree
                _, tree_paths[i] = as_tree_walk(i, instances, labels,
                                                instance_ids, n_instances,
                                                tree_pred, tree_pred_labels,
                                                tree_pred_proba, tree_correct,
                                                feature, threshold, path, features)

        return(batch_paths_container(tree_paths, True))

class batch_getter:

    def __init__(self, instances, labels):
        self.instances = instances
        self.labels = labels
        self.current_row = 0

    def get_next(self, batch_size = 1):
        instances_out = self.instances[self.current_row:self.current_row + batch_size]
        labels_out = self.labels[self.current_row:self.current_row + batch_size]
        self.current_row += batch_size
        return(instances_out, labels_out)

class rule_acc_lite:

    def __init__(self, instance_id, var_dict,
                paths, patterns,
                rule, pruned_rule, conjunction_rule,
                target_class, target_class_label,
                major_class, major_class_label,
                tree_preds, model_post,
                coverage, precision, pri_and_post,
                pri_and_post_accuracy,
                pri_and_post_counts,
                pri_and_post_recall,
                pri_and_post_f1,
                pri_and_post_lift):
        self.instance_id = instance_id
        self.var_dict = var_dict
        self.paths = paths
        self.patterns = patterns
        self.rule = rule
        self.pruned_rule = pruned_rule
        self.conjunction_rule = conjunction_rule
        self.target_class = target_class
        self.target_class_label = target_class_label
        self.major_class = major_class
        self.major_class_label = major_class_label
        self.tree_preds = tree_preds
        self.model_post = model_post
        self.coverage = coverage
        self.precision = precision
        self.pri_and_post = pri_and_post
        self.pri_and_post_accuracy = pri_and_post_accuracy
        self.pri_and_post_counts = pri_and_post_counts
        self.pri_and_post_recall = pri_and_post_recall
        self.pri_and_post_f1 = pri_and_post_f1
        self.pri_and_post_lift = pri_and_post_lift

    def to_dict(self):
        return({'instance_id' : self.instance_id,
        'var_dict' : self.var_dict,
        'paths' : self.paths,
        'patterns' : self.patterns,
        'rule' : self.rule,
        'pruned_rule' : self.pruned_rule,
        'conjunction_rule' : self.conjunction_rule,
        'target_class' :self.target_class,
        'target_class_label' :self.target_class_label,
        'major_class' : self.major_class,
        'major_class_label' :self.major_class_label,
        'tree_preds' : self.tree_preds,
        'model_post' : self.model_post,
        'coverage' : self.coverage,
        'precision' : self.precision,
        'pri_and_post' : self.pri_and_post,
        'pri_and_post_accuracy' : self.pri_and_post_accuracy,
        'pri_and_post_counts' : self.pri_and_post_counts,
        'pri_and_post_recall' : self.pri_and_post_recall,
        'pri_and_post_f1' : self.pri_and_post_f1,
        'pri_and_post_lift' : self.pri_and_post_lift})

    def to_dict(self):
        return([self.instance_id, self.var_dict, self.paths, self.patterns,
                self.rule, self.pruned_rule, self.conjunction_rule,
                self.target_class, self.target_class_label,
                self.major_class, self.major_class_label,
                self.tree_preds, self.model_post,
                self.coverage, self.precision, self.pri_and_post,
                self.pri_and_post_accuracy,
                self.pri_and_post_counts,
                self.pri_and_post_recall,
                self.pri_and_post_f1,
                self.pri_and_post_lift])

class loo_encoder:

    def __init__(self, sample_instances, sample_labels, encoder=None):
        self.sample_instances = sample_instances
        self.sample_labels = sample_labels
        if encoder is None:
            self.encoder = default_encoder
        else:
            self.encoder = encoder

    # leave one out by instance_id and encode the rest
    def loo_encode(self, instance_id):
        instances = self.sample_instances.drop(instance_id)
        labels = self.sample_labels.drop(instance_id)
        enc_instances = self.encoder.transform(instances)
        return(instances, enc_instances, labels)

class rule_evaluator:

    def encode_pred(self, prediction_model, instances=None, bootstrap=False, random_state=None):
        if instances is None:
            pred_instances=self.sample_instances
        else:
            pred_instances=instances
        if random_state is None:
            random_state = self.random_state
        if bootstrap:
            pred_instances = pred_instances.sample(frac=1.0, replace=True, random_state=random_state)
        pred_labels = Series(prediction_model.predict((pred_instances)), index=pred_instances.index)
        return(pred_instances, pred_labels)

    def apply_rule(self, rule=None, instances=None):
        if rule is None:
            rule = self.rule
        if instances is None:
            instances = self.sample_instances
        lt_gt = lambda x, y, z : x < y if z else x > y # if z is True, x < y else x > y
        idx = np.full(instances.shape[0], 1, dtype='bool')
        for r in rule:
            idx = np.logical_and(idx, lt_gt(instances.getcol(self.onehot_features.index(r[0])).toarray().flatten(), r[2], r[1]))
        return(idx)

    def evaluate_rule(self, rule=None, instances=None,
                    labels=None, class_names=None):
        if rule is None:
            rule = self.rule
        if instances is None:
            instances = self.sample_instances
        if labels is None:
            labels = self.sample_labels
        if labels is None:
            print('Test labels are required for rule evaluation')
            return()
        if class_names is None:
            class_names = self.class_names

        idx = self.apply_rule(rule, instances)
        coverage = idx.sum()/len(idx) # tp + fp / tp + fp + tn + fn

        priors = p_count_corrected(labels, [i for i in range(len(class_names))])

        p_counts = p_count_corrected(labels.iloc[idx], [i for i in range(len(class_names))]) # true positives
        post = p_counts['p_counts']
        p_corrected = np.array([p if p > 0.0 else 1.0 for p in post]) # to avoid div by zeros

        counts = p_counts['counts']
        labels = p_counts['labels']

        observed = np.array((counts, priors['counts']))
        if counts.sum() > 0: # previous_counts.sum() == 0 is impossible
            chisq = chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)
        else:
            chisq = None

        # class coverage, TPR (recall) TP / (TP + FN)
        recall = counts / priors['counts']
        r_corrected = np.array([r if r > 0.0 else 1.0 for r in recall]) # to avoid div by zeros
        f1 = [2] * ((post * recall) / (p_corrected + r_corrected))

        not_covered_counts = counts + (np.sum(priors['counts']) - priors['counts']) - (np.sum(counts) - counts)
        # accuracy = (TP + TN) / num_instances formula: https://books.google.co.uk/books?id=ubzZDQAAQBAJ&pg=PR75&lpg=PR75&dq=rule+precision+and+coverage&source=bl&ots=Aa4Gj7fh5g&sig=6OsF3y4Kyk9KlN08OPQfkZCuZOc&hl=en&sa=X&ved=0ahUKEwjM06aW2brZAhWCIsAKHY5sA4kQ6AEIUjAE#v=onepage&q=rule%20precision%20and%20coverage&f=false
        accu = not_covered_counts/priors['counts'].sum()

        # to avoid div by zeros
        pri_corrected = np.array([pri if pri > 0.0 else 1.0 for pri in priors['p_counts']])
        pos_corrected = np.array([pos if pri > 0.0 else 0.0 for pri, pos in zip(priors['p_counts'], post)])
        if counts.sum() == 0:
            rec_corrected = np.array([0.0] * len(pos_corrected))
            cov_corrected = np.array([1.0] * len(pos_corrected))
        else:
            rec_corrected = counts / counts.sum()
            cov_corrected = np.array([counts.sum() / priors['counts'].sum()])

        # lift = precis / (total_cover * prior)
        lift = pos_corrected / ( ( cov_corrected ) * pri_corrected )

        return({'coverage' : coverage,
                'priors' : priors,
                'post' : post,
                'counts' : counts,
                'labels' : labels,
                'recall' : recall,
                'f1' : f1,
                'accuracy' : accu,
                'lift' : lift,
                'chisq' : chisq})

class rule_tester(rule_evaluator):

    def __init__(self, data_container, rule, sample_instances, sample_labels=None):
        self.onehot_features = data_container.onehot_features
        if data_container.class_col in data_container.le_dict.keys():
            self.class_names = data_container.get_label(data_container.class_col, [i for i in range(len(data_container.class_names))])
            self.get_label = data_container.get_label
        else:
            self.class_names = data_container.class_names
            self.get_label = None
        self.rule = rule
        self.sample_instances = sample_instances
        self.sample_labels = sample_labels
        self.random_state = data_container.random_state

class rule_accumulator(rule_evaluator):

    def __init__(self, data_container, paths_container):

        self.instance_id = paths_container.instance_id
        self.random_state = data_container.random_state
        self.onehot_features = data_container.onehot_features
        self.onehot_dict = data_container.onehot_dict
        self.var_dict = deepcopy(data_container.var_dict)
        self.paths = paths_container.paths
        self.patterns = paths_container.patterns
        self.unapplied_rules = [i for i in range(len(self.patterns))]

        self.class_col = data_container.class_col
        if data_container.class_col in data_container.le_dict.keys():
            self.class_names = data_container.get_label(data_container.class_col, [i for i in range(len(data_container.class_names))])
            self.get_label = data_container.get_label
        else:
            self.class_names = data_container.class_names
            self.get_label = None

        self.model_votes = p_count_corrected(paths_container.tree_preds, self.class_names)

        for item in self.var_dict:
            if self.var_dict[item]['class_col']:
                continue
            else:
                if self.var_dict[item]['data_type'] == 'nominal':
                    n_labs = len(self.var_dict[item]['labels'])
                else:
                    n_labs = 1
                self.var_dict[item]['upper_bound'] = [math.inf] * n_labs
                self.var_dict[item]['lower_bound'] = [-math.inf] * n_labs
        self.rule = []
        self.pruned_rule = []
        self.conjunction_rule = []
        self.previous_rule = []
        self.reverted = []
        self.total_points = sum([scrs[2] for scrs in self.patterns])
        self.accumulated_points = 0
        self.encoder = None
        self.sample_instances = None
        self.sample_labels = None
        self.n_instances = None
        self.n_classes = None
        self.target_class = None
        self.target_class_label = None
        self.major_class = None
        self.model_entropy = None
        self.model_info_gain = None
        self.model_post = None
        self.max_ent = None
        self.coverage = None
        self.precision = None
        self.cum_info_gain = None
        self.information_gain = None
        self.prior_entropy = None
        self.prior_info = None
        self.pri_and_post = None
        self.pri_and_post_accuracy = None
        self.pri_and_post_counts = None
        self.pri_and_post_recall = None
        self.pri_and_post_f1 = None
        self.pri_and_post_lift = None
        self.isolation_pos = None
        self.stopping_param = None
        self.build_rule_iter = None

    def add_rule(self, p_total = 0.1):
        self.previous_rule = deepcopy(self.rule)
        next_rule = self.patterns[self.unapplied_rules[0]]
        for item in next_rule[0]:
            if item in self.rule:
                continue # skip duplicates (essential for pruning reasons)
            if item[0] in self.onehot_dict: # binary feature
                # update the master list
                position = self.var_dict[self.onehot_dict[item[0]]]['onehot_labels'].index(item[0])
                if item[1]: # leq_threshold True
                    self.var_dict[self.onehot_dict[item[0]]]['upper_bound'][position] = item[2]
                else:
                    self.var_dict[self.onehot_dict[item[0]]]['lower_bound'][position] = item[2]
                # append or update
                self.rule.append(item)

            else: # continuous feature
                append_or_update = False
                if item[1]: # leq_threshold True
                    if item[2] <= self.var_dict[item[0]]['upper_bound'][0]:
                        self.var_dict[item[0]]['upper_bound'][0] = item[2]
                        append_or_update = True

                else:
                    if item[2] > self.var_dict[item[0]]['lower_bound'][0]:
                        self.var_dict[item[0]]['lower_bound'][0] = item[2]
                        append_or_update = True

                if append_or_update:
                    feature_appears = [(f, ) for (f, t, _) in self.rule]
                    if (item[0],) in feature_appears:
                        # print(item, 'feature appears already')
                        valueless_rule = [(f, t) for (f, t, _) in self.rule]
                        if (item[0], item[1]) in valueless_rule: # it's already there and needs updating
                            # print(item, 'feature values appears already')
                            self.rule[valueless_rule.index((item[0], item[1]))] = item
                        else: # feature has been used at the opposite end (either lower or upper bound) and needs inserting
                            # print(item, 'feature values with new discontinuity')
                            self.rule.insert(feature_appears.index((item[0],)) + 1, item)
                    else:
                        # print(item, 'feature first added')
                        self.rule.append(item)

            # accumlate points from rule and tidy up
            # remove the first item from unapplied_rules as it's just been applied or ignored for being out of range
            self.accumulated_points += self.patterns[0][2]
            del self.unapplied_rules[0]
            # accumlate all the freq patts that are subsets of the current rules
            # remove the index from the unapplied rules list (including the current rule just added)
            to_remove = []
            for ur in self.unapplied_rules:
                # check if all items are already part of the rule (i.e. it's a subset)
                if all([item in self.rule for item in self.patterns[ur][0]]):
                    self.accumulated_points += self.patterns[ur][2]
                    # collect up the values to remove. don't want to edit the iterator in progress
                    to_remove.append(ur)
            for rmv in reversed(to_remove):
                self.unapplied_rules.remove(rmv)

    def prune_rule(self):
        # remove all other binary items if one Greater than is found.
        gt_items = {} # find all the items with the leq_threshold False
        for item in self.rule:
            if ~item[1] and item[0] in self.onehot_dict: # item is greater than thresh and a nominal type
                gt_items[self.onehot_dict[item[0]]] = item[0] # capture the parent feature and the feature value

        gt_pruned_rule = []
        for item in self.rule:
            if item[0] in self.onehot_dict: # binary variable
                if self.onehot_dict[item[0]] not in gt_items.keys():
                    gt_pruned_rule.append(item)
                elif ~item[1]:
                    gt_pruned_rule.append(item)
            else: # leave continuous as is
                gt_pruned_rule.append(item)

        # if all but one of a feature set is False, swap them out for the remaining value
        # start by counting all the lt thresholds in each parent feature
        lt_items = defaultdict(lambda: 0)
        for item in gt_pruned_rule: # find all the items with the leq_threshold True
            if item[1] and item[0] in self.onehot_dict: # item is less than thresh and a nominal type
                lt_items[self.onehot_dict[item[0]]] += 1 # capture the parent feature and count each

        # checking if just one other feature value remains unused
        pruned_items = [item[0] for item in gt_pruned_rule]
        for lt in dict(lt_items).keys():
            n_categories = len([i for i in self.onehot_dict.values() if i == lt])
            if n_categories - dict(lt_items)[lt] == 1:
                # get the remaining value for this feature
                lt_labels = self.var_dict[lt]['onehot_labels']
                to_remove = [label for label in lt_labels if label in pruned_items]
                remaining_value = [label for label in lt_labels if label not in pruned_items]

                lt_pruned_rule = []
                pos = -1
                for rule in gt_pruned_rule:
                    pos += 1
                    if rule[0] not in to_remove:
                        lt_pruned_rule.append(rule)
                    else:
                        # set the position of the last term of the parent feature
                        insert_pos = pos
                        pos -= 1
                lt_pruned_rule.insert(insert_pos, (remaining_value[0], False, 0.5))
                # the main rule is updated for passing through the loop again
                gt_pruned_rule = lt_pruned_rule.copy()

        self.pruned_rule = gt_pruned_rule

        # find a rule with only binary True values
        self.conjunction_rule = [r for r in self.pruned_rule if ~r[1]]

    def __greedy_commit__(self, current, previous):
        if current <= previous:
            self.rule = deepcopy(self.previous_rule)
            self.reverted.append(True)
            return(True)
        else:
            self.reverted.append(False)
            return(False)

    def build_rule(self, encoder, sample_instances, sample_labels, prediction_model
                        , stopping_param = 1
                        , precis_threshold = 1.0
                        , fixed_length = None
                        , target_class=None
                        , greedy=None
                        , bootstrap=False
                        , random_state=None):

        # basic setup
        if random_state is None:
            random_state=self.random_state
        if stopping_param > 1 or stopping_param < 0:
            stopping_param = 1
            print('warning: stopping_param should be 0 <= p <= 1. Value was reset to 1')
        self.stopping_param = stopping_param
        self.encoder = encoder
        self.sample_instances = sample_instances
        self.sample_labels = sample_labels
        self.n_classes = len(np.unique(self.sample_labels))
        self.n_instances = len(self.sample_labels)

        # model posterior
        # model votes collected in constructor
        self.model_post = self.model_votes['p_counts']

        # model final entropy
        self.model_entropy = entropy(self.model_post)

        # model predicted class
        self.major_class = np.argmax(self.model_post)
        if self.get_label is None:
            self.major_class_label = self.major_class
        else:
            self.major_class_label = self.get_label(self.class_col, self.major_class)

        # this analysis
        # target class
        if target_class is None:
            self.target_class = self.major_class
            self.target_class_label = self.major_class_label
        else:
            self.target_class = target_class
            if self.get_label is None:
                self.target_class_label = self.target_class
            else:
                self.target_class_label = self.get_label(self.class_col, self.target_class)

        # first get all the predictions from the model
        pred_instances, pred_labels = self.encode_pred(prediction_model, sample_instances, bootstrap=bootstrap, random_state=random_state) # what the model would predict on the training sample

        # prior - empty rule
        p_counts = p_count_corrected(pred_labels.values, [i for i in range(len(self.class_names))])
        self.pri_and_post = np.array([p_counts['p_counts'].tolist()])
        self.pri_and_post_counts = np.array([p_counts['counts'].tolist()])
        self.pri_and_post_recall = [np.full(self.n_classes, 1.0)] # counts / prior counts
        self.pri_and_post_f1 =  [2] * ( ( self.pri_and_post * self.pri_and_post_recall ) / ( self.pri_and_post + self.pri_and_post_recall ) ) # 2 * (precis * recall/(precis + recall) )
        self.pri_and_post_accuracy = np.array([p_counts['p_counts'].tolist()])
        self.pri_and_post_lift = [np.full(self.n_classes, 1.0)] # precis / (total_cover * prior)
        self.prior_entropy = entropy(self.pri_and_post_counts[0])

        # info gain
        self.max_ent = entropy([1 / self.n_classes] * self.n_classes)
        self.model_info_gain = self.max_ent - self.model_entropy
        self.prior_info = self.max_ent - self.prior_entropy

        # pre-loop set up
        # rule based measures - prior/empty rule
        current_precision = p_counts['p_counts'][np.where(p_counts['labels'] == self.target_class)][0] # based on priors

        self.coverage = [1.0]
        self.precision = [current_precision]

        # rule posteriors
        previous_entropy = self.max_ent # start at max possible
        current_entropy = self.prior_entropy # entropy of prior distribution
        self.information_gain = [previous_entropy - current_entropy] # information baseline (gain of priors over maximum)
        self.cum_info_gain = self.information_gain.copy()

        # accumulate rule terms
        cum_points = 0
        self.build_rule_iter = 0

        while current_precision != 1.0 and current_precision != 0.0 and current_precision < precis_threshold and self.accumulated_points <= self.total_points * self.stopping_param and (fixed_length is None or len(self.cum_info_gain) < max(1, fixed_length) + 1):
            self.build_rule_iter += 1
            self.add_rule(p_total = self.stopping_param)
            # you could add a round of bootstrapping here, but what does that do to performance
            eval_rule = self.evaluate_rule(instances=encoder.transform(pred_instances),
                                            labels=pred_labels)

            # entropy / information
            previous_entropy = current_entropy
            current_entropy = entropy(eval_rule['post'])

            # code to confirm rule, or revert to previous
            # choosing from a range of possible metrics and learning improvement
            # possible to introduce annealing?
            # e.g if there was no change, or an decrease in precis
            if greedy is not None:
                if greedy == 'precision':
                    current = eval_rule['post'][np.where(eval_rule['labels'] == self.target_class)]
                    previous = list(reversed(self.pri_and_post))[0][np.where(eval_rule['labels'] == self.target_class)]
                    should_continue = self.__greedy_commit__(current, previous)
                elif greedy == 'f1':
                    current = eval_rule['f1'][np.where(eval_rule['labels'] == self.target_class)]
                    previous = list(reversed(self.pri_and_post_f1))[0][np.where(eval_rule['labels'] == self.target_class)]
                    should_continue = self.__greedy_commit__(current, previous)
                elif greedy == 'accuracy':
                    current = eval_rule['accuracy'][np.where(eval_rule['labels'] == self.target_class)]
                    previous = list(reversed(self.pri_and_post_accuracy))[0][np.where(eval_rule['labels'] == self.target_class)]
                    should_continue = self.__greedy_commit__(current, previous)
                elif greedy == 'chi2':
                    previous_counts = list(reversed(self.pri_and_post_counts))[0]
                    observed = np.array((eval_rule['counts'], previous_counts))
                    if eval_rule['counts'].sum() == 0: # previous_counts.sum() == 0 is impossible
                        should_continue = self.__greedy_commit__(1, 0) # go ahead with rule as the algorithm will finish here
                    else: # do the chi square test but mask any classes where prev and current are zero
                        should_continue = self.__greedy_commit__(0.05, chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)[1])
                # add more options here
                else: should_continue = False
                if should_continue:
                    continue # don't update all the metrics, just go to the next round

            # check for end conditions; no target class coverage
            if eval_rule['post'][np.where(eval_rule['labels'] == self.target_class)] == 0.0:
                current_precision = 0.0
            else:
                current_precision = eval_rule['post'][np.where(eval_rule['labels'] == self.target_class)][0]

            # if we keep the new rule, append the results to the persisted arrays
            # general coverage and precision
            self.precision.append(current_precision)
            self.coverage.append(eval_rule['coverage'])

            # per class measures
            self.pri_and_post = np.append(self.pri_and_post, [eval_rule['post']], axis=0)
            self.pri_and_post_counts = np.append(self.pri_and_post_counts, [eval_rule['counts']], axis=0)
            self.pri_and_post_accuracy = np.append(self.pri_and_post_accuracy, [eval_rule['accuracy']], axis=0)
            self.pri_and_post_recall = np.append(self.pri_and_post_recall, [eval_rule['recall']], axis=0 )
            self.pri_and_post_f1 = np.append(self.pri_and_post_f1, [eval_rule['f1']], axis=0 )
            self.pri_and_post_lift = np.append(self.pri_and_post_lift, [eval_rule['lift']], axis=0 )

            # entropy and info gain
            self.information_gain.append(previous_entropy - current_entropy)
            self.cum_info_gain.append(sum(self.information_gain))

        # first time major_class is isolated
        if any(np.argmax(self.pri_and_post, axis=1) == self.target_class):
            self.isolation_pos = np.min(np.where(np.argmax(self.pri_and_post, axis=1) == self.target_class))
        else: self.isolation_pos = None

    def score_rule(self, alpha=0.5):
        target_precision = [p[self.target_class] for p in self.pri_and_post]
        target_recall = [r[self.target_class] for r in self.pri_and_post_recall]
        target_f1 = [f[self.target_class] for f in self.pri_and_post_f1]
        target_accuracy = [a[self.target_class] for a in self.pri_and_post_accuracy]
        target_prf = [[p, r, f, a] for p, r, f, a in zip(target_precision, target_recall, target_f1, target_accuracy)]

        target_cardinality = [i for i in range(len(target_precision))]

        lf = lambda x: math.log2(x + 1)
        score_fun1 = lambda f, crd, alp: lf(f * crd * alp / (1.0 + ((1 - alp) * crd**2)))
        score_fun2 = lambda a, crd, alp: lf(a * crd * alp / (1.0 + ((1 - alp) * crd**2)))

        score1 = [s for s in map(score_fun1, target_f1, target_cardinality, [alpha] * len(target_cardinality))]
        score2 = [s for s in map(score_fun2, target_accuracy, target_cardinality, [alpha] * len(target_cardinality))]

        return(target_prf, score1, score2)

    def lite_instance(self):
        return(rule_acc_lite(self.instance_id, self.var_dict, self.paths,
        self.patterns, self.rule, self.pruned_rule, self.conjunction_rule,
        self.target_class, self.target_class_label,
        self.major_class, self.major_class_label,
        self.model_votes, self.model_post,
        self.coverage, self.precision, self.pri_and_post,
        self.pri_and_post_accuracy,
        self.pri_and_post_counts,
        self.pri_and_post_recall,
        self.pri_and_post_f1,
        self.pri_and_post_lift))
