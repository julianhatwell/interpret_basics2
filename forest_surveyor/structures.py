import sys
import math
import numpy as np
from forest_surveyor import p_count
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
from forest_surveyor import config as cfg

class data_container:

    def __init__(self
    , data
    , class_col
    , var_names = None
    , var_types = None
    , pickle_dir = ''
    , random_state = 123
    , spiel = ''):
        self.spiel = spiel
        self.random_state = random_state

        self.data = data
        self.data_pre = DataFrame.copy(self.data)
        self.class_col = class_col
        self.pickle_dir = pickle_dir

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

            self.var_dict[v] = {'labels' : names if t == 'nominal' else None
                                , 'onehot_labels' : [v + '_' + str(n) for n in names] if t == 'nominal' else None
                                , 'class_col' : True if v == class_col else False
                                , 'data_type' : t}

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

        if len(self.categorical_features) > 0:
            encoder = OneHotEncoder(categorical_features=self.categorical_features)
            encoder.fit(self.data_pre.as_matrix())
            self.encoder = encoder
        else:
            self.encoder = None

    # helper function for pickling files
    def pickle_path(self, filename = ''):
        if len(cfg.project_dir) > 0:
            return(cfg.project_dir + cfg.path_sep + self.pickle_dir + cfg.path_sep + filename)
        else:
            return(self.pickle_dir + cfg.path_sep + filename)

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
        if random_state is None:
            random_state = self.random_state
        X, y = self.data_pre[self.features], self.data_pre[self.class_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        train_priors = y_train.value_counts().sort_index()/len(y_train)
        test_priors = y_test.value_counts().sort_index()/len(y_test)

        # one hot encoding required for classifier
        # otherwise integer vectors will be treated as ordinal
        # OneHotEncoder takes an integer list as an argument to state which columns to encode
        if self.encoder is not None:
            X_train_enc = self.encoder.transform(X_train)
        else:
            X_train_enc = sparse.csr_matrix(X_train)

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
        return([bin_or_cont(f, t, v, self.onehot_dict) for f, t, v in rule])

class paths_container:
    def __init__(self
    , path_detail
    , by_tree):
        self.by_tree = by_tree
        self.path_detail = path_detail
        self.instance = None
        self.paths = None
        self.patterns = None

    def flip(self):
        n = len(self.path_detail[0])
        flipped_paths = [[]] * n
        for i in range(n):
            flipped_paths[i] =  [p[i] for p in self.path_detail]
        self.path_detail = flipped_paths
        self.by_tree = not self.by_tree

    def major_class_from_paths(self, return_counts=True):
        if self.by_tree:
            pred_classes = [self.path_detail[i][self.instance]['pred_class'] for i in range(len(self.path_detail))]
        else:
            pred_classes = [self.path_detail[self.instance][i]['pred_class'] for i in range(len(self.path_detail[self.instance]))]

        unique, counts = np.unique(pred_classes, return_counts=True)

        if return_counts:
            return(unique[np.argmax(counts)], dict(zip(unique, counts)))
        else: return(unique[np.argmax(counts)])

    def set_paths(self, instance, which_trees='all', feature_values=True):
        # i hate this code. must improve and de-dup!
        self.instance = math.floor(instance) # make sure it's an integer
        true_to_lt = lambda x: '<' if x == True else '>'

        if self.by_tree:
            n_paths = len(self.path_detail)
            if which_trees == 'correct':
                paths_info = [self.path_detail[i][self.instance]['path'] for i in range(n_paths) if self.path_detail[i][self.instance]['tree_correct']]
            elif which_trees == 'majority':
                major_class = self.major_class_from_paths(return_counts=False)
                paths_info = [self.path_detail[i][self.instance]['path'] for i in range(n_paths) if self.path_detail[i][self.instance]['pred_class'] == major_class]
            elif which_trees == 'minority':
                major_class = self.major_class_from_paths(return_counts=False)
                paths_info = [self.path_detail[i][self.instance]['path'] for i in range(n_paths) if self.path_detail[i][self.instance]['pred_class'] != major_class]
            else:
                paths_info = [self.path_detail[i][self.instance]['path'] for i in range(n_paths)]
        else:
            n_paths = len(self.path_detail[self.instance])
            if which_trees == 'correct':
                paths_info = [self.path_detail[self.instance][i]['path'] for i in range(n_paths) if self.path_detail[self.instance][i]['tree_correct']]
            elif which_trees == 'majority':
                major_class = self.major_class_from_paths(return_counts=False)
                paths_info = [self.path_detail[self.instance][i]['path'] for i in range(n_paths) if self.path_detail[self.instance][i]['pred_class'] == major_class]
            elif which_trees == 'minority':
                major_class = self.major_class_from_paths(return_counts=False)
                paths_info = [self.path_detail[self.instance][i]['path'] for i in range(n_paths) if self.path_detail[self.instance][i]['pred_class'] != major_class]
            else:
                paths_info = [self.path_detail[self.instance][i]['path'] for i in range(n_paths)]

        if feature_values:
            paths = [[]] * len(paths_info)
            for i, p in enumerate(paths_info):
                paths[i] = [(f, leq, t) for f, leq, t in zip(p['feature_name'], p['leq_threshold'], p['threshold'])]
        else:
            paths = [p['feature_name'] for p in paths_info]

        self.paths = paths

    def discretize_paths(self, var_dict, bins=4, equal_counts=False):

        # can't continue without a previous run of get_paths()
        if self.instance is not None:
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

    def sort_fp(self, alpha=0.0):
        # alpha can be any number, but best results -1 < alpha < 1
        # negative numbers will favour shorter fp
        # positive numbers will favour longer fp
        alpha = float(alpha)
        fp_scope = freq_patts.copy()
        # to shrink the support of shorter freq_patterns
        # formula is log(sup * (len - alpha) / len)
        score_function = lambda x: (x[0], x[1], math.log(x[1]) * (len(x[0]) - alpha) / len(x[0]))
        fp_scope = [fp for fp in map(score_function, fp_scope.items())]

        # score is now at position 2 of tuple
        return(sorted(fp_scope, key=itemgetter(2), reverse = True))

    def set_patterns(self, support=0.1, alpha=0.0, sort=True):

        # convert to an absolute number of instances rather than a fraction
        if support < 1:
            support = round(support * len(self.paths))
        self.patterns = find_frequent_patterns(self.paths, support)
        if sort:
            alpha = float(alpha)
            fp_scope = self.patterns.copy()
            # to shrink the support of shorter freq_patterns
            # formula is log(sup * (len - alpha) / len)
            score_function = lambda x: (x[0], x[1], math.log(x[1]) * (len(x[0]) - alpha) / len(x[0]))
            fp_scope = [fp for fp in map(score_function, fp_scope.items())]
            # score is now at position 2 of tuple
            self.patterns = sorted(fp_scope, key=itemgetter(2), reverse = True)

class forest_walker:

    def __init__(self
    , forest
    , features
    , encoder = None
    , prediction_model = None):
        self.forest = forest
        self.encoder = encoder
        if prediction_model is None:
            self.prediction_model = forest
        else: self.prediction_model = prediction_model

        self.features = features
        self.n_features = len(self.features)

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

    def tree_walk(self, tree, instances, labels = None, features = None):

        n_instances = instances.shape[0]
        # encode features prior to sending into forest for path analysis
        if self.encoder is None:
            instances = np.matrix(instances)
            n_features = instances.shape[1]
        else:
            instances = self.encoder.transform(instances)
            if 'todense' in dir(instances): # it's a sparse matrix
                instances = instances.todense()
            n_features = instances.shape[1]

        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        tree_pred = tree.predict(instances)
        tree_pred_proba = tree.predict_proba(instances)

        if labels is None:
            tree_correct = [None] * n_instances
        else:
            tree_correct = tree_pred == labels.values

        path = tree.decision_path(instances).indices
        path_deque = deque(path)
        ic = -1 # instance_count
        instance_paths = [{}] * n_instances
        while len(path_deque) > 0:
            p = path_deque.popleft()
            if feature[p] < 0: # leaf node
                continue
            feature_value = instances[ic, [feature[p]]].item(0)
            leq_threshold = feature_value <= threshold[p]
            if features is None:
                feature_name = None
            else:
                feature_name = features[feature[p]]
            if p == 0: # root node
                ic += 1
                if labels is None:
                    true_class = None
                else:
                    true_class = labels.values[ic]
                instance_paths[ic] = { 'pred_class' : tree_pred[ic].astype(np.int64)
                                        , 'pred_proba' : tree_pred_proba[ic].tolist()
                                        , 'true_class' : true_class
                                        , 'tree_correct' : tree_correct[ic]
                                        , 'path' : {'feature_idx' : [feature[p]]
                                                                , 'feature_name' : [feature_name]
                                                                , 'feature_value' : [feature_value]
                                                                , 'threshold' : [threshold[p]]
                                                                , 'leq_threshold' : [leq_threshold]
                                                    }
                                        }
            else:
                instance_paths[ic]['path']['feature_idx'].append(feature[p])
                instance_paths[ic]['path']['feature_name'].append(feature_name)
                instance_paths[ic]['path']['feature_value'].append(feature_value)
                instance_paths[ic]['path']['threshold'].append(threshold[p])
                instance_paths[ic]['path']['leq_threshold'].append(leq_threshold)

        return(instance_paths)

    def forest_walk(self, instances, labels = None):

        tree_paths = [[]] * len(self.forest.estimators_)
        for i, t in enumerate(self.forest.estimators_):
            tree_paths[i] = self.tree_walk(tree = t
                                       , instances = instances
                                       , labels = labels
                                       , features = self.features)

        return(paths_container(tree_paths, True))

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
                target_class, major_class,
                model_votes, model_post,
                coverage, precision, pri_and_post,
                pri_and_post_accuracy,
                pri_and_post_counts,
                pri_and_post_coverage):
        self.instance_id = instance_id
        self.var_dict = var_dict
        self.paths = paths
        self.patterns = patterns
        self.rule = rule
        self.pruned_rule = pruned_rule
        self.conjunction_rule = conjunction_rule
        self.target_class = target_class
        self.major_class = major_class
        self.model_votes = model_votes
        self.model_post = model_post
        self.coverage = coverage
        self.precision = precision
        self.pri_and_post = pri_and_post
        self.pri_and_post_accuracy = pri_and_post_accuracy
        self.pri_and_post_counts = pri_and_post_counts
        self.pri_and_post_coverage = pri_and_post_coverage

    def to_dict(self):
        return({'instance_id' : self.instance_id,
        'var_dict' : self.var_dict,
        'paths' : self.paths,
        'patterns' : self.patterns,
        'rule' : self.rule,
        'pruned_rule' : self.pruned_rule,
        'conjunction_rule' : self.conjunction_rule,
        'target_class' :self.target_class,
        'major_class' : self.major_class,
        'model_votes' : self.model_votes,
        'model_post' : self.model_post,
        'coverage' : self.coverage,
        'precision' : self.precision,
        'pri_and_post' : self.pri_and_post,
        'pri_and_post_accuracy' : self.pri_and_post_accuracy,
        'pri_and_post_counts' : self.pri_and_post_counts,
        'pri_and_post_coverage' : self.pri_and_post_coverage})

    def to_dict(self):
        return([self.instance_id, self.var_dict, self.paths, self.patterns,
                self.rule, self.pruned_rule, self.conjunction_rule,
                self.target_class, self.major_class, self.model_votes, self.model_post,
                self.coverage, self.precision, self.pri_and_post,
                self.pri_and_post_accuracy,
                self.pri_and_post_counts,
                self.pri_and_post_coverage])

class rule_accumulator:

    def __init__(self, data_container, paths_container, instance_id):

        self.instance_id = instance_id
        self.onehot_features = data_container.onehot_features
        self.onehot_dict = data_container.onehot_dict
        self.var_dict = deepcopy(data_container.var_dict)
        self.paths = paths_container.paths
        if paths_container.by_tree:
            self.model_votes = p_count([paths_container.path_detail[t][paths_container.instance]['pred_class'] for t in range(len(paths_container.paths))])
        else:
            self.model_votes = p_count([paths_container.path_detail[paths_container.instance][t]['pred_class'] for t in range(len(paths_container.paths))])
        self.patterns = paths_container.patterns
        self.unapplied_rules = [i for i in range(len(self.patterns))]

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
        self.total_points = sum([scrs[2] for scrs in self.patterns])
        self.accumulated_points = 0
        self.sample_instances = None
        self.sample_labels = None
        self.n_instances = None
        self.n_classes = None
        self.target_class = None
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
        self.pri_and_post_coverage = None
        self.isolation_pos = None
        self.stopping_param = None

    def add_rule(self, p_total = 0.1):
        self.previous_rule = self.rule
        next_rule = self.patterns[self.unapplied_rules[0]]
        for item in next_rule[0]:
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

    def profile(self, sample_instances, sample_labels
                        , stopping_param = 1
                        , fixed_length = None
                        , target_class=None):

        # basic setup
        if stopping_param > 1 or stopping_param < 0:
            stopping_param = 1
            print('warning: stopping_param should be 0 <= p <= 1. Value was reset to 1')
        self.stopping_param = stopping_param
        self.sample_instances = sample_instances
        self.sample_labels = sample_labels
        self.n_classes = len(np.unique(self.sample_labels))
        self.n_instances = len(self.sample_labels)

        # model posterior - model votes collected in constructor
        self.major_class = np.argmax(self.model_votes['p_counts'])
        self.model_post = np.empty(self.n_classes)
        for cn in range(self.n_classes):
            if cn in self.model_votes['labels']:
                self.model_post[cn] = self.model_votes['p_counts'][np.where(self.model_votes['labels'] == cn)][0]
            else:
                self.model_post[cn] = 0.0

        # model final entropy
        self.model_entropy = entropy(self.model_post)

        if target_class is None: target_class = self.major_class
        self.target_class = target_class

        # prior
        p_counts = p_count(sample_labels.values)
        self.pri_and_post = [p_counts['p_counts'].tolist()]
        self.pri_and_post_counts = [p_counts['counts'].tolist()]
        self.pri_and_post_coverage = [np.full(self.n_classes, 1.0)]
        self.pri_and_post_accuracy = [p_counts['p_counts'].tolist()]
        self.prior_entropy = entropy(self.pri_and_post_counts[0])

        # info gain
        self.max_ent = entropy([1 / self.n_classes] * self.n_classes)
        self.model_info_gain = self.max_ent - self.model_entropy
        self.prior_info = self.max_ent - self.prior_entropy

        # pre-loop set up
        # rule based measures - prior/empty rule
        current_precision = p_counts['p_counts'][np.where(p_counts['labels'] == self.target_class)][0] # based on priors
        self.coverage = [1]
        self.precision = [current_precision]

        # rule posteriors
        previous_entropy = self.max_ent # start at max possible
        current_entropy = self.prior_entropy # entropy of prior distribution
        self.information_gain = [previous_entropy - current_entropy] # information baseline (gain of priors over maximum)
        self.cum_info_gain = self.information_gain.copy()

        # accumulate rule terms
        cum_points = 0
        while current_precision != 1.0 and self.accumulated_points <= self.total_points * self.stopping_param and (fixed_length is None or len(self.cum_info_gain) < fixed_length + 1):
            self.add_rule(p_total = self.stopping_param)
            p_counts = p_count(sample_labels.loc[self.apply_rule()].values)

            # code to confirm rule, or revert to previous can go here
            # choosing from a range of possible metrics and a learning improvement
            # possible to introduce annealing?

            if np.shape(p_counts['p_counts'][np.where(p_counts['labels'] == self.target_class)])[0] <= 0:
                current_precision = 1.0
            else:
                current_precision = p_counts['p_counts'][np.where(p_counts['labels'] == self.target_class)][0] # based on priors

            # general coverage and precision
            self.precision.append(current_precision)
            n_coverage = sum(p_counts['counts'])
            self.coverage.append(n_coverage/self.n_instances)

            # posterior distributions and counts
            post = np.empty(self.n_classes)
            counts = np.empty(self.n_classes)
            not_covered_counts = np.empty(self.n_classes)
            # per class (not all classes are represented and this needs to be tested each iteration)
            for cn in range(self.n_classes):
                if cn in p_counts['labels']:
                    post[cn] = p_counts['p_counts'][np.where(p_counts['labels'] == cn)][0]
                    counts[cn] = p_counts['counts'][np.where(p_counts['labels'] == cn)][0]
                    not_covered_counts[cn] = counts[cn] + (np.sum(self.pri_and_post_counts[0]) - self.pri_and_post_counts[0][cn]) - (np.sum(p_counts['counts']) - counts[cn])
                else:
                    post[cn] = 0.0
                    counts[cn] = 0.0
                    not_covered_counts[cn] = 0.0

            # class coverage, TPR
            # accuracy (TP + TN) / num_instances formula: https://books.google.co.uk/books?id=ubzZDQAAQBAJ&pg=PR75&lpg=PR75&dq=rule+precision+and+coverage&source=bl&ots=Aa4Gj7fh5g&sig=6OsF3y4Kyk9KlN08OPQfkZCuZOc&hl=en&sa=X&ved=0ahUKEwjM06aW2brZAhWCIsAKHY5sA4kQ6AEIUjAE#v=onepage&q=rule%20precision%20and%20coverage&f=false
            self.pri_and_post_coverage.append(counts/self.pri_and_post_counts[0])
            self.pri_and_post_accuracy.append(not_covered_counts/self.n_instances)

            # append the results to the array
            previous_entropy = current_entropy
            current_entropy = entropy(post)
            self.information_gain.append(previous_entropy - current_entropy)
            self.cum_info_gain.append(sum(self.information_gain))
            self.pri_and_post = np.append(self.pri_and_post, [post], axis=0)
            self.pri_and_post_counts = np.append(self.pri_and_post_counts, [counts], axis=0)

        # first time major_class is isolated
        if any(np.argmax(self.pri_and_post, axis=1) == self.target_class):
            self.isolation_pos = np.min(np.where(np.argmax(self.pri_and_post, axis=1) == self.target_class))
        else: self.isolation_pos = None

    def score_rule(self, alpha=0.5):
        target_precision = [p[self.target_class] for p in self.pri_and_post]
        target_coverage = [c[self.target_class] for c in self.pri_and_post_coverage]
        target_accuracy = [a[self.target_class] for a in self.pri_and_post_accuracy]
        target_pca = [[p, c, a] for p, c, a in zip(target_precision, target_coverage, target_accuracy)]

        target_cardinality = [i for i in range(len(target_precision))]

        lf = lambda x: math.log2(x + 1)
        score_fun1 = lambda p, c, crd, alp: alp * p + (1.0 - alp) * c * crd / (1.0 + crd**2)
        score_fun2 = lambda a, crd: lf(a * crd / (1.0 + crd**2))

        score1 = [s for s in map(score_fun1, target_precision, target_coverage, target_cardinality, [alpha] * len(target_precision))]
        score2 = [s for s in map(score_fun2, target_accuracy, target_cardinality)]

        return(target_pca, score1, score2)

    def lite_instance(self):
        return(rule_acc_lite(self.instance_id, self.var_dict, self.paths,
        self.patterns, self.rule, self.pruned_rule,
        self.conjunction_rule, self.target_class, self.major_class,
        self.model_votes, self.model_post,
        self.coverage, self.precision, self.pri_and_post,
        self.pri_and_post_accuracy,
        self.pri_and_post_counts,
        self.pri_and_post_coverage))
