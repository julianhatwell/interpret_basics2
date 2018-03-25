import sys
import numpy as np
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import deque
from scipy.stats import sem
from itertools import chain

if sys.platform == 'win32':
    path_sep = '\\'
else:
    path_sep = '/'

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
            self.var_names =list(self.data.columns)
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
        X_train_enc = self.encoder.transform(X_train)

        return({
        'X_train': X_train,
        'X_train_enc' : X_train_enc,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'encoder' : self.encoder})

class forest_paths_container:
    def __init__(self
    , forest_paths
    , by_tree):
        self.by_tree = by_tree
        self.forest_paths = forest_paths

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

    # helper function to encode features prior to sending into forest for path analysis
    def enc_features(self, instances, feature_encoding):
        if feature_encoding is None:
            n_features = instances.shape[1]
            addendum = "\n"
        else:
            instances = feature_encoding.transform(instances)
            if 'todense' in dir(instances): # it's a sparse matrix
                instances = instances.todense()
            n_features = instances.shape[1]
            addendum = "(after encoding)\n"
        return(instances, n_features, addendum)

    def tree_walk(self, tree, instances, labels = None, features = None):

        n_instances = instances.shape[0]
        instances, n_features, addendum = self.enc_features(instances, self.encoder)

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

    def forest_walk(self, instances, labels = None, by_tree=True):
        # TO DO: as above with possiblity to flip between by_tree=True and by_tree=False?

        tree_paths = [[]] * len(self.forest.estimators_)
        for i, t in enumerate(self.forest.estimators_):
            tree_paths[i] = self.tree_walk(tree = t
                                       , instances = instances
                                       , labels = labels
                                       , features = self.features)
        if by_tree:
            forest_paths = forest_paths_container(tree_paths, by_tree)
        else:
            n_instances = instances.count()[0]
            instance_paths = [[]] * n_instances
            for i in range(n_instances):
                instance_paths[i] =  [tp[i] for tp in tree_paths]
            forest_paths = forest_paths_container(instance_paths, by_tree)

        return(forest_paths)

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
