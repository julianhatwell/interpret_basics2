from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import pydotplus
from IPython.display import Image
import math

from operator import itemgetter
from collections import deque, defaultdict
from itertools import combinations, product
from copy import deepcopy
from cycler import cycler

import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
from sklearn.externals.six import StringIO
from sklearn.pipeline import make_pipeline
from scipy.stats import itemfreq, sem, entropy, rankdata
from pandas.tools.plotting import parallel_coordinates
from matplotlib.ticker import MaxNLocator

# helper function for returning counts and proportions of unique values in an array
def p_count(arr):
    labels, counts = np.unique(arr, return_counts = True)
    return(
    {'labels' : labels,
    'counts' : counts,
    'p_counts' : counts / len(arr)})

# helper for plotting varimp
def plot_feature_stats(statistics, stat_names, stat_name, feature_names):
    n = len(statistics)

    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    x_pos = [i for i in range(len(feature_names))]
    fig, ax = plt.subplots(1, 1, figsize=(11,3))

    stms = []
    for s, stat in enumerate(statistics):
        linefmt = 'C' + str(s) + '-'
        markerfmt = 'C' + str(s) + 'o'

        stm = ax.stem(x_pos, stat[0], markerfmt=markerfmt, linefmt=linefmt)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation='vertical')
        ax.set_title(stat_name)
        x_pos = [i + 0.2 for i in x_pos]
        stms.append(stm)

    plt.legend(stms, stat_names)
    plt.show()

def plot_varimp(model, feature_names, ordered=False):
    fig, ax = plt.subplots(1, 1, figsize=(11,3))
    x_pos = range(len(feature_names))
    if ordered:
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    else:
        imp = pd.Series(model.feature_importances_, index=feature_names)
    ax.stem(x_pos, imp)
    ax.set_ylabel('Importance %')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(imp.index, rotation='vertical')
    ax.set_title('Variable Importance Plot')
    plt.show()

# helper function for plotting conf mat
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# helper function for plotting comparison mean path lengths from a forest stats dictionary
# TO DO standard error bars
def plot_mean_path_lengths(explore_forest_stats, class_labels=None):

    classes = [c for c in explore_forest_stats]
    mean_path_lengths = np.zeros(len(classes))
    sd_path_lengths = np.zeros(len(classes))

    for i, c in enumerate(classes):
        mean_path_lengths[i] = explore_forest_stats[c]['m_path_length']
        sd_path_lengths[i] = explore_forest_stats[c]['sd_path_length']

    if class_labels:
        classes[:len(class_labels)] = class_labels

    plt.bar(range(len(classes)),
            mean_path_lengths,
            yerr=sd_path_lengths,
            tick_label=classes)
    plt.title('Mean and St.Dev of Decision Path Length by Class')
    plt.ylabel('Root-to-leaf length (number of nodes)')
    plt.show()

# helper function to plot correct vs incorrect tree stats
def cor_incor_plot(cor, incor, label):

    def fit_cor_incor(cor, incor, ax, label, statistic, st_error, resids = False):
        x = cor[label][statistic]
        y = incor[label][statistic]
        xerr = cor[label][st_error]
        yerr = incor[label][st_error]
        # unity line
        fit = np.polyfit(x, x, deg=1)
        ax.plot(x, fit[0] * x + fit[1], color='red', lw = 0.5)
        ax.legend(labels=['y=x'])
        # actual points
        ax.errorbar(x, y, xerr = xerr, yerr = yerr, fmt='.')
        ax.set_xlabel('correct trees')
        ax.set_ylabel('incorrect trees')
        ax.set_title(statistic)

        return(y - x)

    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(9, 6), dpi=100, facecolor='w', edgecolor='k')

    fit_cor_incor(cor, incor, ax = ax1, label = label, statistic = 'm_feature_depth', st_error = 'se_feature_depth')
    fit_cor_incor(cor, incor, ax = ax2, label = label, statistic = 'm_root_traversals', st_error = 'se_root_traversals')
    fit_cor_incor(cor, incor, ax = ax3, label = label, statistic = 'm_child_traversals', st_error = 'se_child_traversals')
    resids = fit_cor_incor(cor, incor, ax = ax4, label = label
                           , statistic = 'm_lower_traversals'
                           , st_error = 'se_lower_traversals'
                           , resids = True)
    fit_cor_incor(cor, incor, ax = ax5, label = label, statistic = 'm_nonroot_traversals', st_error = 'se_nonroot_traversals')
    fit_cor_incor(cor, incor, ax = ax6, label = label, statistic = 'm_node_traversals', st_error = 'se_node_traversals')

    fig.tight_layout()
    plt.show()

    return(resids, rankdata(abs(resids)))

# helper function to plot feature importance vs tree stats
def stat_impor_plot(importances, statistics, label):

    def fit_stat_impor(importances, statistic, ax, label):
        ax.scatter(importances, statistics[label][statistic], s=5)
        ax.set_xlabel('feature importance')
        ax.set_ylabel(statistic)

    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(9, 6), dpi=100, facecolor='w', edgecolor='k')

    fit_stat_impor(importances, statistic = 'n_node_traversals', ax = ax1, label = label)
    fit_stat_impor(importances, statistic = 'p_root_traversals', ax = ax2, label = label)
    fit_stat_impor(importances, statistic = 'p_child_traversals', ax = ax3, label = label)
    fit_stat_impor(importances, statistic = 'p_lower_traversals', ax = ax4, label = label)
    fit_stat_impor(importances, statistic = 'p_nonroot_traversals', ax = ax5, label = label)
    fit_stat_impor(importances, statistic = 'm_feature_depth', ax = ax6, label = label)

    fig.suptitle('feature importances vs structural properties', fontsize=16)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.show()

# helper plot for viewing differences between feature usage
# combining st errs by division: https://chem.libretexts.org/Core/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error
def log_ratio_plot(num, denom, labels, num_err=None, denom_err=None, top=3):
    fig, ax = plt.subplots(figsize=(11, 3))
    log_ratio = np.log(num/denom)

    top_n = np.flip(np.argpartition(log_ratio, -top)[-top:], axis=0)
    bot_n = np.flip(np.argpartition(-log_ratio, -top)[-top:], axis=0)

    lr_top = [log_ratio[i] for i in top_n]
    lr_bot = [log_ratio[i] for i in bot_n]

    if num_err is not None and denom_err is not None:
        ax.stem(top_n, lr_top, linefmt = 'C' + str(1) + ':', markerfmt = 'C' + str(1) + '.')
        ax.stem(bot_n, lr_bot, linefmt = 'C' + str(2) + ':', markerfmt = 'C' + str(2) + '.')

        yerr = 0.434*np.sqrt((num_err/num)**2 + (denom_err/denom)**2)
        ax.errorbar(range(len(labels)), log_ratio, yerr = yerr, fmt='o')
    else:
        ax.stem(range(len(labels)), log_ratio)
        ax.stem(top_n, lr_top, linefmt = 'C' + str(1) + '-', markerfmt = 'C' + str(1) + 'o')
        ax.stem(bot_n, lr_bot, linefmt = 'C' + str(2) + '-', markerfmt = 'C' + str(2) + 'o')

    ax.axhline(0.0, color = 'k', ls = '--')
    ax.annotate('1:1', xy=(-1.0, max(log_ratio) * 0.1))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('log(ratio)')

    plt.show()
    if num_err is not None:
        return(log_ratio, yerr)
    else:
        return(log_ratio)

# can't remember where this came from
# helper function for plotting a tree model
def plot_tree_inline(tree, class_names=None, feature_names=None):
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data
                , class_names=class_names
                , feature_names=feature_names
                , filled=True, rounded=True
                , special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return(Image(graph.create_png()))

# from http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
# def get_tree_state(tree, feature_names=None, class_names=None):
#     major_class = [np.argmax(vals[0]) for vals in tree.__getstate__()['values']]
#     if feature_names is not None:
#         feature_nodes = []
#         for f in tree.feature:
#             if f < 0:
#                 feature_nodes.append("leaf")
#             else:
#                 feature_nodes.append(feature_names[f])
#     else:
#         feature_nodes = None
#     if class_names is not None:
#         class_nodes = []
#         for c in major_class:
#             class_nodes.append(class_names[c])
#     else:
#         class_nodes = None
#     return(major_class, feature_nodes, class_nodes)

# helper function used in tree stats analysis
def enc_features(instances, feature_encoding):
    if feature_encoding is None:
        n_features = instances.shape[1]
        addendum = "\n"
    else:
        instances = feature_encoding.transform(instances)
        n_features = instances.shape[1]
        addendum = "(after encoding)\n"
    return(instances, n_features, addendum)

def explore_forest(forest
                    , instances
                    , labels
                    , feature_encoding = None):
                    # , return_counts = False): # what's return_counts again?

    n_instances = instances.shape[0]
    if labels is not None:
        if len(labels) != n_instances:
            raise ValueError("labels and instances must be same length")

    trees = forest.estimators_
    n_trees = len(trees)

    instances, n_features, addendum = enc_features(instances, feature_encoding)

    print("Number of trees in this forest:", n_trees)
    print("Number features in this data batch:", n_features, addendum)

    # base counts for all trees
    root_features = np.zeros(n_features) # set up a 1d feature array to count features appearing as root nodes
    child_features = np.zeros(n_features)
    lower_features = np.zeros(n_features)

    # TO DO: move this into the lower loop, with a counter to ensure it only happens the first time
    # That will save an iteration through the whole forest
    for i, t in enumerate(trees):
        tree = t.tree_
        feature = tree.feature
        children_left = tree.children_left
        children_right = tree.children_right

        root_features[feature[0]] += 1
        if children_left[0] >= 0:
            child_features[feature[children_left[0]]] +=1
        if children_right[0] >= 0:
            child_features[feature[children_right[0]]] +=1

        for i, f in enumerate(feature):
            if i < 3: continue # root and children
            if f < 0: continue # leaf nodes
            lower_features[f] += 1
    # will collect stats for each class label, as it passes each feature node in each tree
    tree_features = {}
    for lab in np.unique(labels):
        # enumerate() won't skip a number if missing so need to get the id explicitly
        idx = labels == lab
        instances_lab = instances[idx.values] # instance ids having the current label
        n_instances_lab = instances_lab.shape[0]

        tree_features_lab = np.zeros(n_features) # set up a 1d feature array to count the feature nodes
        feature_depth = np.full(n_features, np.nan) # set up a 1d feature array for counting
        tree_performance = np.full(n_trees, np.nan) # set up a 1d feature array for tree performance
        child_features_lab = np.zeros(n_features) # array to track number instances in first level nodes
        lower_features_lab = np.zeros(n_features) # array to track number instances in lower level nodes
        path_lengths_lab = np.zeros((n_instances_lab, n_trees))
        print("Number of instances for class", lab, "in this data batch:", n_instances_lab)

        # work through each tree
        for i, t in enumerate(trees):
            tree = t.tree_
            feature = tree.feature
            tree_pred = t.predict(instances_lab) # predictions for current class
            tree_performance[i] = sum(tree_pred == lab)

            path = t.decision_path(instances_lab).indices
            paths_begin = np.where(path == 0)
            paths_end = np.append(np.where(path == 0)[0][1:], len(path))
            path_lengths_lab[:,i] = paths_end - paths_begin

            counts_ft = np.unique(path, return_counts = True)[1]
            features_traversed = feature[np.unique(path, return_counts = True)[0]]
            features_traversed_count = sorted(zip(features_traversed, counts_ft), key = lambda x: x[0])

            # accumulate the number of instances that traverse each feature.
            # don't count leaf nodes
            for ftc in features_traversed_count:
                if ftc[0] >=0:
                    tree_features_lab[ftc[0]] += ftc[1]

            # depth counting
            # create a deque (iterable)
            path_deque = deque(path)
            depth = 0
            while len(path_deque) > 0:
                p = path_deque.popleft()
                if feature[p] < 0: # leaf node
                    continue
                if p == 0: # root node
                    depth = 0
                else:
                    depth += 1
                    if depth == 1: # child node
                        child_features_lab[feature[p]] += 1
                    if depth > 1: # lower node
                        lower_features_lab[feature[p]] += 1
                if np.isnan(feature_depth[feature[p]]):
                     feature_depth[feature[p]] = depth
                else:
                    feature_depth[feature[p]] += depth

        root_features_lab = root_features * n_instances_lab
        nonroot_features_lab = tree_features_lab - root_features_lab
        tree_features[lab] = {'n_instances' : n_instances_lab # number of instances in class / following stats separate by class
                              , 'n_node_traversals' : tree_features_lab # number of times feature node was reached
                              , 'n_root_traversals' : root_features_lab # number of time feature was a root node
                              , 'n_nonroot_traversals' : nonroot_features_lab # number of time feature was a nonroot node
                              , 'n_child_traversals' : child_features_lab # number of times feature was a child of root node
                              , 'n_lower_traversals' : lower_features_lab # number of times feature was lower than a child of root node
                              , 'p_node_traversals' : tree_features_lab/np.sum(tree_features_lab) # proportion of total times...
                              , 'p_root_traversals' : root_features_lab/np.sum(root_features_lab)
                              , 'p_nonroot_traversals' : nonroot_features_lab/np.sum(nonroot_features_lab)
                              , 'p_child_traversals' : child_features_lab/np.sum(child_features_lab)
                              , 'p_lower_traversals' : lower_features_lab/np.sum(lower_features_lab)
                              , 'm_node_traversals' : tree_features_lab/(n_instances_lab * n_trees) # mean number of times... over all instances and trees
                              , 'm_root_traversals' : root_features_lab/(n_instances_lab * n_trees)
                              , 'm_nonroot_traversals' : nonroot_features_lab/(n_instances_lab * n_trees)
                              , 'm_child_traversals' : child_features_lab/(n_instances_lab * n_trees)
                              , 'm_lower_traversals' : lower_features_lab/(n_instances_lab * n_trees)
                              , 'n_correct_preds' : np.sum(tree_performance) # total number of correct predictions
                              , 'm_correct_preds' : np.mean(tree_performance) # TO DO: fix/check this, it doesn't look right. mean number of correct predictions (out of all trees in the forest per instance)
                              # TO DO: safe coding for sd and se when sample size is very small (minority class cases)
                              #, 'sd_correct_preds' : np.std(tree_performance, ddof=1) # st.dev of correct predictions (out of all trees in the forest per instance)
                              #, 'sem_correct_preds' : sem(tree_performance) # se mean of correct predictions (out of all trees in the forest per instance)
                              , 'p_correct_preds' : np.mean(tree_performance)/n_instances_lab # proportion of correct predictions over all trees and instances
                              , 'feature_depth' : feature_depth
                              , 'm_feature_depth' : feature_depth/tree_features_lab
                              , 'm_path_length' : np.mean(path_lengths_lab)
                              #, 'sd_path_length' : np.std(path_lengths_lab, ddof=1) # mean number of correct predictions (out of all trees in the forest per instance)
                              #, 'sem_path_length' : path_lengths_lab # mean number of correct predictions (out of all trees in the forest per instance)
                              }

    all_classes_nodes = np.zeros(n_features)
    all_classes_correct_preds = 0
    all_classes_feature_depth = np.zeros(n_features)
    all_classes_roots = np.zeros(n_features)
    all_classes_nonroots = np.zeros(n_features)
    all_classes_children = np.zeros(n_features)
    all_classes_lower = np.zeros(n_features)
    for tf in tree_features:
        all_classes_nodes += tree_features[tf]['n_node_traversals']
        all_classes_nonroots += tree_features[tf]['n_nonroot_traversals']
        all_classes_roots += tree_features[tf]['n_root_traversals']
        all_classes_children += tree_features[tf]['n_child_traversals']
        all_classes_lower += tree_features[tf]['n_lower_traversals']
        all_classes_correct_preds += tree_features[tf]['n_correct_preds']
        all_classes_feature_depth = tree_features[tf]['m_feature_depth'] * tree_features[tf]['n_instances'] / n_instances

    all_classes_path_length = 0
    for lab in np.unique(labels):
        all_classes_path_length += tree_features[lab]['m_path_length'] * tree_features[lab]['n_instances']

    tree_features['all_classes'] = {'n_instances' : n_instances
                                    , 'n_node_traversals' : all_classes_nodes
                                    , 'n_root_traversals' : all_classes_roots
                                    , 'n_nonroot_traversals' : all_classes_nonroots
                                    , 'n_child_traversals' : all_classes_children
                                    , 'n_lower_traversals' : all_classes_lower
                                    , 'p_node_traversals' : all_classes_nodes / np.sum(all_classes_nodes)
                                    , 'p_root_traversals' : all_classes_roots / np.sum(all_classes_roots)
                                    , 'p_nonroot_traversals' : all_classes_nonroots / np.sum(all_classes_nonroots)
                                    , 'p_child_traversals' : all_classes_children / np.sum(all_classes_children)
                                    , 'p_lower_traversals' : all_classes_lower / np.sum(all_classes_lower)
                                    , 'm_node_traversals' : all_classes_nodes / (n_instances * n_trees)
                                    , 'm_root_traversals' : all_classes_roots / (n_instances * n_trees)
                                    , 'm_nonroot_traversals' : all_classes_nonroots / (n_instances * n_trees)
                                    , 'm_child_traversals' : all_classes_children / (n_instances * n_trees)
                                    , 'm_lower_traversals' : all_classes_lower / (n_instances * n_trees)
                                    , 'n_correct_preds' : all_classes_correct_preds
                                    , 'm_correct_preds' : all_classes_correct_preds / n_trees
                                    , 'p_correct_preds' : all_classes_correct_preds / (n_trees * n_instances)
                                    , 'm_feature_depth' : all_classes_feature_depth
                                    , 'root_features' : root_features
                                    , 'child_features' : child_features
                                    , 'lower_features' : lower_features
                                    , 'm_path_length' : all_classes_path_length / n_instances
                                    }

    return(tree_features)

# to replace explore_forest()
class forest_surveyor:

    def __init__(self
    , model
    , features
    , prediction_model = None):
        self.model = model
        if prediction_model is None:
            self.prediction_model = model
        else: self.prediction_model = prediction_model

        # base counts for all trees
        self.root_features = np.zeros(len(features)) # set up a 1d feature array to count features appearing as root nodes
        self.child_features = np.zeros(len(features))
        self.lower_features = np.zeros(len(features))
        self.structure = {'root_features' : self.root_features
                         , 'child_features' : self.child_features
                         , 'lower_features' : self.lower_features}

        # walk through each tree to get the structure
        for t, tree in enumerate(self.model.estimators_):

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

    def fit(self
        , instances
        , labels
        , features):

        self.instances = instances
        self.labels = labels
        self.features = features

        self.n_instances = instances.shape[0]
        self.n_features = len(features)
        self.n_classes = len(np.unique(labels))

        if labels is not None:
            if len(labels) != self.n_instances:
                raise ValueError("labels and instances must be same length")

        trees = self.model.estimators_
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
            idx = pd.Series([True] * self.n_instances) # it's easier if has the same type as the labels
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

def tree_path(tree, instances, labels = None, feature_encoding = None, feature_names = None):

    n_instances = instances.shape[0]
    instances, n_features, addendum = enc_features(instances, feature_encoding)

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
        feature_value = instances.todense()[ic, [feature[p]]].item(0)
        leq_threshold = feature_value <= threshold[p]
        if feature_names is None:
            feature_name = None
        else:
            feature_name = feature_names[feature[p]]
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

def forest_path(forest, instances, labels = None, feature_encoding = None, feature_names = None, by_tree=True):
    # TO DO: define a class that has a self attribute determining whether organinsed by tree or by instance.
    # TO DO: as above with possiblity to flip between?
    # TO DO: some of the functions like get_paths() can then belong to the class

    tree_paths = [[]] * len(forest.estimators_)
    for i, t in enumerate(forest.estimators_):
        tree_paths[i] = tree_path(tree = t
                                   , instances = instances
                                   , labels = labels
                                   , feature_encoding = feature_encoding
                                   , feature_names = feature_names)

    if by_tree:
        return(tree_paths)

    n_instances = instances.count()[0]
    instance_paths = [[]] * n_instances
    for i in range(n_instances):
        instance_paths[i] =  [tp[i] for tp in tree_paths]
    return(instance_paths)

def major_class_from_forest_paths(forest_paths, instance, by_tree=True, return_counts=True):
    if by_tree:
        pred_classes = [forest_paths[i][instance]['pred_class'] for i in range(len(forest_paths))]
    else:
        pred_classes = [forest_paths[instance][i]['pred_class'] for i in range(len(forest_paths[instance]))]

    unique, counts = np.unique(pred_classes, return_counts=True)

    if return_counts:
        return(unique[np.argmax(counts)], dict(zip(unique, counts)))
    else: return(unique[np.argmax(counts)])

# function to extract all the paths for one instance
# from a forest_paths object created by the forest_path function
def get_paths(forest_paths, instance, by_tree=True, which_trees='all', feature_values=True):
    # i hate this code. must improve and de-dup!
    # TO DO: by_tree / by_instance should be based on a class attribute
    instance == math.floor(instance) # make sure it's an integer
    true_to_lt = lambda x: '<' if x == True else '>'

    if by_tree:
        n_paths = len(forest_paths)
        if which_trees == 'correct':
            paths_info = [forest_paths[i][instance]['path'] for i in range(n_paths) if forest_paths[i][instance]['tree_correct']]
        elif which_trees == 'majority':
            major_class = major_class_from_forest_paths(forest_paths, instance, by_tree, return_counts=False)
            paths_info = [forest_paths[i][instance]['path'] for i in range(n_paths) if forest_paths[i][instance]['pred_class'] == major_class]
        elif which_trees == 'minority':
            major_class = major_class_from_forest_paths(forest_paths, instance, by_tree, return_counts=False)
            paths_info = [forest_paths[i][instance]['path'] for i in range(n_paths) if forest_paths[i][instance]['pred_class'] != major_class]
        else:
            paths_info = [forest_paths[i][instance]['path'] for i in range(n_paths)]
    else:
        n_paths = len(forest_paths[instance])
        if which_trees == 'correct':
            paths_info = [forest_paths[instance][i]['path'] for i in range(n_paths) if forest_paths[instance][i]['tree_correct']]
        elif which_trees == 'majority':
            major_class = major_class_from_forest_paths(forest_paths, instance, by_tree, return_counts=False)
            paths_info = [forest_paths[instance][i]['path'] for i in range(n_paths) if forest_paths[instance][i]['pred_class'] == major_class]
        else:
            paths_info = [forest_paths[instance][i]['path'] for i in range(n_paths)]

    if feature_values:
        paths = [[]] * len(paths_info)
        for i, p in enumerate(paths_info):
            #paths[i] = [f + true_to_lt(leq) + str(t) for f, leq, t in zip(p['feature_name'], p['leq_threshold'], p['threshold'])]
            paths[i] = [(f, leq, t) for f, leq, t in zip(p['feature_name'], p['leq_threshold'], p['threshold'])]
    else:
        paths = [p['feature_name'] for p in paths_info]
    return(paths)

# from https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "\t" * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

# get plain labels back from onehot_features names
def decode_onehot(label, colnames):
    candidate_matches = [cn for cn in colnames if cn in label]
    # hopefully only one found
    if len(candidate_matches) == 1:
        return(candidate_matches[0])
    # otherwise check the head of the string (if finds no match, wil return nothing)
    else:
        for cm in candidate_matches:
            if cm in label[0:len(cm)]: return(cm)

# get plain labels back from onehot_features names for object returned by get_paths()
def decode_onehot_paths(paths, labels, condense=True):
    # loop through each path and item, update as going along
    # maybe a better algo would find all the positions of each option and update them all at once
    for i, path in enumerate(paths):
        for j, p in enumerate(path):
            paths[i][j] = decode_onehot(p, labels)
        if condense:
            paths[i] = list(set(paths[i]))
    return(paths)

def apriori(transactions, support, max_itemset_size):

    def clear_items(candidate_dct, support, pass_nbr):

        # Only keep new patterns whose support is higher than threshold
        to_be_del = [i for i, c in candidate_dct.items() if c < support]
        for tbd in to_be_del:
            del candidate_dct[tbd]

        # Determine which patterns were added to the list since previous pass
        if pass_nbr > 1:
            new_freq_patts = [k for k in candidate_dct.keys() if len(k) == pass_nbr]
        else: new_freq_patts = []

        if new_freq_patts: # Python check, will skip if list is empty
            # Delete patterns where there is a new superset
            for nfp in new_freq_patts:
                open_sets_keys = [k for k in candidate_dct.keys()
                             if set(nfp).issuperset(set(k)) and len(k) < pass_nbr]
                for osk in sorted(open_sets_keys, reverse=True):
                    del candidate_dct[osk]

        return(candidate_dct)

    # find candidates in dataset
    def data_pass(transactions, support, pass_nbr, candidate_dct):
        if pass_nbr==1:
            # just add a one-item tuple for each item, counting how often it appears
            for line in transactions:
                for item in line:
                    candidate_dct[(item,)]+=1

        elif pass_nbr==2:
            frequent_items = [ck[0] for ck in candidate_dct.keys()]
            for line in transactions:
                # reject items that are not frequent before combining remaining items in the line
                line = [item for item in line if item in frequent_items]
                # then add a tuple to the candidate_dct
                for item_set in combinations(sorted(line), pass_nbr):
                    candidate_dct[item_set]+=1

        else:
            # work out what are the next longest candidate frequent items
            frequent_items = [c for c in candidate_dct.keys() if len(c) == pass_nbr - 1]
            head = ''
            candidates = []
            for c in sorted(frequent_items):
                # check, if not new head (== TRUE)
                # create a new candidate of length plus 1
                # by adding the last element of the current
                # to the previous head
                if head == c[:pass_nbr - 2]:
                    candidates.append(sum((prev, c[pass_nbr - 2:pass_nbr - 1]), ()))
                # otherwise, just set a new head
                else:
                    prev = c
                    head = c[:pass_nbr - 2]

            for line in transactions:
                # only consider sets that appear in the candidates list and count them
                for item_set in combinations(sorted(line), pass_nbr):
                    if item_set in candidates: candidate_dct[item_set]+=1

        candidate_dct = clear_items(candidate_dct, support, pass_nbr)

        return(candidate_dct)

    # convert to an absolute number of instances rather than a fraction
    if support < 1:
        support = round(support * len(transactions))

    candidate_dct = defaultdict(lambda: 0)
    prev_len = -1
    for i in range(max_itemset_size):
        candidate_dct = data_pass(transactions,
                                  support,
                                  pass_nbr=i+1,
                                  candidate_dct=candidate_dct)
        # stop if no further frequent item sets found
        if len(candidate_dct) == prev_len:
            print('No frequent patterns longer than', i, 'items. Stopping early.')
            break
        prev_len = len(candidate_dct)

    return dict(candidate_dct)


def discretize_paths(paths, vars_dict, bins):

    cont_vars = [vn for vn in vars_dict if vars_dict[vn]['data_type'] == 'continuous' and vars_dict[vn]['class_col'] == False]
    for feature in cont_vars:
        # lower bound, greater than
        lowers = [item[2] for nodes in paths for item in nodes if item[0] == feature and item[1] == False]
        # upper bound, less than
        uppers = [item[2] for nodes in paths for item in nodes if item[0] == feature and item[1] == True]

        upper_bins = np.histogram(uppers, bins=bins)[1]
        lower_bins = np.histogram(lowers, bins=bins)[1]

        # upper_bin_midpoints = pd.Series(upper_bins).rolling(window=2, center=False).mean().values[1:]
        upper_bin_means = (np.histogram(uppers, upper_bins, weights=uppers)[0] /
                            np.histogram(uppers, upper_bins)[0]).round(2)

        # lower_bin_midpoints = pd.Series(lower_bins).rolling(window=2, center=False).mean().values[1:]
        lower_bin_means = (np.histogram(lowers, lower_bins, weights=lowers)[0] /
                            np.histogram(lowers, lower_bins)[0]).round(2)

        # discretize functions from histogram means
        upper_discretize = lambda x: upper_bin_means[np.max([np.min([np.digitize(x, upper_bins), len(upper_bin_means)]), 1]) - 1]
        lower_discretize = lambda x: lower_bin_means[np.max([np.min([np.digitize(x, lower_bins, right= True), len(upper_bin_means)]), 1]) - 1]


        paths_discretized = []
        for nodes in paths:
            nodes_discretized = []
            for f, t, v in nodes:
                if f == feature:
                    if t == False: # greater than, lower bound
                        v = lower_discretize(v)
                    else:
                        v = upper_discretize(v)
                nodes_discretized.append((f, t, v))
            paths_discretized.append(nodes_discretized)
        # at the end of each loop, update the original
        paths = paths_discretized

    # return fully discretized continuous features
    return(paths)

def sort_fp(freq_patts, alpha=0.0):
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

def get_sorted_fp(paths, support = 0.1, max_itemset_size = 5, alpha = 0.0):
    freq_patt = apriori(transactions = paths, support = support, max_itemset_size = max_itemset_size)
    sorted_patt = sort_fp(freq_patt, alpha=alpha)
    return(sorted_patt)

class rule_accumulator:

    def __init__(self, vars_dict, onehot_dict, rule_list):
        # TO DO: make scored_list (of rules) a class? Output of sort_fp function, which could be part of fp object?
        # TO DO: then check type on init
        self.onehot_dict = onehot_dict
        self.vars_list = deepcopy(vars_dict)
        self.rule_list = rule_list
        self.unapplied_rules = [i for i in range(len(self.rule_list))]
        for item in self.vars_list:
            if self.vars_list[item]['class_col']:
                continue
            else:
                if self.vars_list[item]['data_type'] == 'nominal':
                    n_labs = len(self.vars_list[item]['labels'])
                else:
                    n_labs = 1
                self.vars_list[item]['upper_bound'] = [math.inf] * n_labs
                self.vars_list[item]['lower_bound'] = [-math.inf] * n_labs
        self.rule = []

        self.total_points = sum([scrs[2] for scrs in self.rule_list])
        self.accumulated_points = 0

    def add_rule(self, p_total = 0.1):
        next_rule = self.rule_list[self.unapplied_rules[0]]
        for item in next_rule[0]:
            if item[0] in self.onehot_dict: # binary feature
                # update the master list
                position = self.vars_list[self.onehot_dict[item[0]]]['onehot_labels'].index(item[0])
                if item[1]: # leq_threshold True
                    self.vars_list[self.onehot_dict[item[0]]]['upper_bound'][position] = item[2]
                else:
                    self.vars_list[self.onehot_dict[item[0]]]['lower_bound'][position] = item[2]
                # append or update
                self.rule.append(item)

            else: # continuous feature
                append_or_update = False
                if item[1]: # leq_threshold True
                    if item[2] <= self.vars_list[item[0]]['upper_bound'][0]:
                        self.vars_list[item[0]]['upper_bound'][0] = item[2]
                        append_or_update = True

                else:
                    if item[2] > self.vars_list[item[0]]['lower_bound'][0]:
                        self.vars_list[item[0]]['lower_bound'][0] = item[2]
                        append_or_update = True

                if append_or_update:
                    feature_appears = [(f, ) for (f, t, _) in self.rule]
                    if (item[0],) in feature_appears:
                        print(item, 'feature appears already')
                        valueless_rule = [(f, t) for (f, t, _) in self.rule]
                        if (item[0], item[1]) in valueless_rule: # it's already there and needs updating
                            print(item, 'feature values appears already')
                            self.rule[valueless_rule.index((item[0], item[1]))] = item
                        else: # feature has been used at the opposite end (either lower or upper bound) and needs inserting
                            print(item, 'feature values with new discontinuity')
                            self.rule.insert(feature_appears.index((item[0],)) + 1, item)
                    else:
                        print(item, 'feature first added')
                        self.rule.append(item)

            # accumlate points from rule
            # remove the first item from unapplied_rules as it's just been applied or ignored for being out of range
            self.accumulated_points += self.rule_list[0][2]
            del self.unapplied_rules[0]
            # accumlate all the freq patts that are subsets of the current rules
            # remove the index from the unapplied rules list (including the current rule just added)
            to_remove = []
            for ur in self.unapplied_rules:
                # check if all items are already part of the rule (i.e. it's a subset)
                if all([item in self.rule for item in self.rule_list[ur][0]]):
                    self.accumulated_points += self.rule_list[ur][2]
                    # collect up the values to remove. don't want to edit the iterator in progress
                    to_remove.append(ur)
            for rmv in reversed(to_remove):
                self.unapplied_rules.remove(rmv)

        return(self.rule, self.accumulated_points)

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
                lt_labels = self.vars_list[lt]['onehot_labels']
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
        self.tight_rule = [r for r in self.pruned_rule if ~r[1]]

def apply_rule(rule, data, features):
    lt_gt = lambda x, y, z : x < y if z else x > y # if z is True, x < y else x > y
    idx = np.full(data.shape[0], 1, dtype='bool')
    for r in rule:
        idx = np.logical_and(idx, lt_gt(data.getcol(features.index(r[0])).toarray().flatten(), r[2], r[1]))
    return(idx)

def pretty_print(rule_list, onehot_dict):
    Tr_Fa = lambda x, y, z : x + ' True' if ~y else x + ' False'
    lt_gt = lambda x, y, z : x + ' <= ' + str(z) if y else x + ' > ' + str(z)
    def bin_cont(x, y, z, onehot_dict):
        if x in onehot_dict:
            return(Tr_Fa(x,y,z))
        else:
            return(lt_gt(x,y,z))
    return([bin_cont(f, t, v, onehot_dict) for f, t, v in rule_list])

# find the short rule that isolates an instance
# find the short rule that isolates an instance
def profile_instance(instance
                        , fp_object, rule_acc
                        , training_data, training_labels
                        , features, class_names
                        , stopping_param = 1
                        , target_class=None):

    # basic setup
    if stopping_param > 1 or stopping_param < 0:
        stopping_param = 1
        print('warning: stopping_param should be 0 <= p <= 1. Reset to value = 1')
    n_trees = len(fp_object[instance]) # to do - check if by_trees or by_instance and adjust accordingly
    n_classes = len(np.unique(training_labels))
    n_instances = len(training_labels)

    # model posterior
    model_votes = p_count([fp_object[instance][t]['pred_class'] for t in range(n_trees)])
    major_class = np.argmax(model_votes['p_counts'])
    model_post = np.empty(n_classes)
    for cn in range(n_classes):
        if cn in model_votes['labels']:
            model_post[cn] = model_votes['p_counts'][np.where(model_votes['labels'] == cn)][0]
        else:
            model_post[cn] = 0.0

    # model final entropy
    model_entropy = entropy(model_post)

    if target_class is None: target_class = major_class

    # prior
    p_counts = p_count(training_labels.values)
    pri_and_post = [p_counts['p_counts'].tolist()]
    pri_and_post_counts = [p_counts['counts'].tolist()]
    pri_and_post_coverage = [np.full(n_classes, 1.0)]
    pri_and_post_accuracy = [p_counts['p_counts'].tolist()]
    prior_entropy = entropy(pri_and_post_counts[0])

    # info gain
    max_ent = entropy([1 / n_classes] * n_classes)
    model_info_gain = max_ent - model_entropy
    prior_info = max_ent - prior_entropy

    # pre-loop set up
    # rule based measures - prior/empty rule
    current_precision = p_counts['p_counts'][np.where(p_counts['labels'] == target_class)][0] # based on priors
    coverage = [1]
    precision = [current_precision]

    # rule posteriors
    previous_entropy = max_ent # start at max possible
    current_entropy = prior_entropy # entropy of prior distribution
    information_gain = [previous_entropy - current_entropy] # information baseline (gain of priors over maximum)
    cum_info_gain = information_gain.copy()

    # accumulate rule terms
    cum_points = 0
    while current_precision != 1.0 and rule_acc.accumulated_points <= rule_acc.total_points * stopping_param:
        rule_acc.add_rule(p_total = stopping_param)
        p_counts = p_count(training_labels.loc[apply_rule(rule_acc.rule, training_data, features)].values)

        if np.shape(p_counts['p_counts'][np.where(p_counts['labels'] == target_class)])[0] <= 0:
            current_precision = 1.0
        else:
            current_precision = p_counts['p_counts'][np.where(p_counts['labels'] == target_class)][0] # based on priors

        # general coverage and precision
        precision.append(current_precision)
        n_coverage = sum(p_counts['counts'])
        coverage.append(n_coverage/n_instances)

        # posterior distributions and counts
        post = np.empty(n_classes)
        counts = np.empty(n_classes)
        not_covered_counts = np.empty(n_classes)
        # per class (not all classes are represented and this needs to be tested each iteration)
        for cn in range(n_classes):
            if cn in p_counts['labels']:
                post[cn] = p_counts['p_counts'][np.where(p_counts['labels'] == cn)][0]
                counts[cn] = p_counts['counts'][np.where(p_counts['labels'] == cn)][0]
                not_covered_counts[cn] = counts[cn] + (np.sum(pri_and_post_counts[0]) - pri_and_post_counts[0][cn]) - (np.sum(p_counts['counts']) - counts[cn])
            else:
                post[cn] = 0.0
                counts[cn] = 0.0
                not_covered_counts[cn] = 0.0

        # class coverage
        # accuracy formula: https://books.google.co.uk/books?id=ubzZDQAAQBAJ&pg=PR75&lpg=PR75&dq=rule+precision+and+coverage&source=bl&ots=Aa4Gj7fh5g&sig=6OsF3y4Kyk9KlN08OPQfkZCuZOc&hl=en&sa=X&ved=0ahUKEwjM06aW2brZAhWCIsAKHY5sA4kQ6AEIUjAE#v=onepage&q=rule%20precision%20and%20coverage&f=false
        pri_and_post_coverage.append(counts/pri_and_post_counts[0])
        pri_and_post_accuracy.append(not_covered_counts/n_instances)

        # append the results to the array
        previous_entropy = current_entropy
        current_entropy = entropy(post)
        information_gain.append(previous_entropy - current_entropy)
        cum_info_gain.append(sum(information_gain))
        pri_and_post = np.append(pri_and_post, [post], axis=0)
        pri_and_post_counts = np.append(pri_and_post_counts, [counts], axis=0)

    # first time major_class is isolated
    if any(np.argmax(pri_and_post, axis=1) == target_class):
        isolation_pos = np.min(np.where(np.argmax(pri_and_post, axis=1) == target_class))
    else: isolation_pos = None

    def resize_plot(ax, set_legend = True):
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        if set_legend:
            ax.legend(title = 'classes', labels = class_names
            , loc='center left', bbox_to_anchor=(1, 0.5))
        return(ax)

    def add_maj_match(ax, isolation_pos):
        height_just = math.floor(ax.get_ylim()[1])
        if isolation_pos is None:
            ax.annotate('target class not matched',
                        xy=(0, height_just * 0.9))
        else:
            ax.axvline(isolation_pos, color = '0.75', ls = '--')
            ax.annotate('target class matched',
                        xy=(isolation_pos + 0.1, height_just * 0.9))
        return(ax)

    def trace_covprecis_plot(ax, measures, measure):
        ax.plot(measures)
        ax.set_title('Rule ' + measure)
        ax.set_ylabel(measure)
        ax.set_xlabel('number of terms')
        return(ax)

    # plot the rule trace based on entropy and posteriors
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), dpi=80
                                    , facecolor='w', edgecolor='k')

    ax1.plot(cum_info_gain)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.axhline(max_ent, color = 'k', ls = '--')
    ax1.axhline(model_info_gain, color = '0.5', ls = '--')
    ax1.axhline(prior_info, color = '0.75', ls = '--')
    ax1.set_title('IG per term added to rule')
    ax1.set_ylabel('Cum. Info. Gain')
    right_just = math.floor(ax1.get_xlim()[1])
    ax1.annotate('max entropy for ' + str(n_classes) + ' class problem',
                xy=(0.5, max_ent - 0.15))
    ax1.annotate('model info gain for inst. ' + str(instance),
                xy=(right_just / 2, model_info_gain - 0.15))
    ax1.annotate('prior information',
                xy=(right_just - 2, prior_info + 0.05))

    ax2.plot(pri_and_post)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim(0.0, 1.0)
    ax2 = add_maj_match(ax2, isolation_pos)
    ax2.set_title('Posterior distributions')
    ax2.set_ylabel('P(y = class)')

    ax3.plot(np.log(pri_and_post_counts + 1))
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3 = add_maj_match(ax3, isolation_pos)
    ax3.set_title('Number of instances')
    ax3.set_ylabel('log(counts)')

    ax1 = resize_plot(ax1, set_legend=False)
    ax2 = resize_plot(ax2)
    ax3 = resize_plot(ax3)

    fig.suptitle('Rule trace through training data', fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    plt.show()

    # plot the rule trace based on coverage and precision
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), dpi=80
                                    , facecolor='w', edgecolor='k')

    ax1 = trace_covprecis_plot(ax1, coverage, 'Coverage')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = trace_covprecis_plot(ax2, pri_and_post_coverage, 'Coverage wrt Class')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = add_maj_match(ax2, isolation_pos)
    ax2 = resize_plot(ax2)
    ax3 = trace_covprecis_plot(ax3, pri_and_post_accuracy, 'Accuracy wrt Class')
    ax3.set_ylim(0.0, 1.0)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3 = add_maj_match(ax3, isolation_pos)
    ax3 = resize_plot(ax3)

    fig.suptitle('Rule trace through training data', fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    plt.show()

    return(rule_acc, model_votes)

# code for OOB sampling
if True:
    '''
    from sklearn.ensemble.forest import _generate_unsampled_indices
    # X here - training set of examples
    n_samples = X.shape[0]
    for tree in rf.estimators_:
        # Here at each iteration we obtain out of bag samples for every tree.
        unsampled_indices = _generate_unsampled_indices(
        tree.random_state, n_samples)
    '''

# not sure if will use this - feature reachability matrix
if True:
    '''
    k_len = max([len(ohk) for k in vars_dict.keys() for ohk in vars_dict[k]['labels']])
    feature_reachability_matrix = np.full((best_params['n_estimators'], len(features)), '', dtype='<U' + str(k_len))

    for t, path in enumerate(paths_0):
        features_visited = []
        for p in path:
            if p[0] in features_visited:
                continue
            else:
                features_visited.append(p[0])
                feature_reachability_matrix[t][np.where(np.array(features) == onehot_dict[p[0]])[0][0]] = \
                vars_dict[features[np.where(np.array(features) == onehot_dict[p[0]])[0][0]]]['labels'][np.where(np.array(vars_dict[features[np.where(np.array(features) == onehot_dict[p[0]])[0][0]]]['onehot_labels']) == p[0])[0][0]]

    # the problem is it can only hold one value for features visited, choosing the first
    # can be checked for true or false
    print(feature_reachability_matrix[400])
    print()

    k_len = max([len(ohk) for k in vars_dict.keys() for ohk in vars_dict[k]['onehot_labels']])
    feature_value_reachability_matrix = np.full((best_params['n_estimators'], len(onehot_features)), '', dtype='<U' + str(k_len))

    for t, path in enumerate(paths_0):
        features_visited = []
        for p in path:
            if p[0] in features_visited:
                continue
            else:
                features_visited.append(p[0])
                feature_value_reachability_matrix[t][np.where(np.array(onehot_features) == p[0])[0][0]] = ~p[1]
                # TO DO: for numerical case, False has to convert to < and True to >=

    # the problem is it can only hold one value for features visited, choosing the first
    print(feature_value_reachability_matrix[400])
    '''

# Code for arbitrarily swapping a feature within a tree
if True:
    '''
    swap_val_1 = 0
    swap_val_2 = 9
    swap_locs_1 = []
    swap_locs_2 = []
    for i, n in enumerate(rf.estimators_[0].tree_.feature):
        if n == swap_val_1: swap_locs_1.append(i) # locations to replace value 1
        elif n == swap_val_2: swap_locs_2.append(i) # locations to replace value 2
        else: continue # of no interest

    rf.estimators_[0].tree_.feature[swap_locs_1] = swap_val_2
    rf.estimators_[0].tree_.feature[swap_locs_2] = swap_val_1
    '''

print('''Utility code in the associated file performs the following steps:
defines function to print pretty confusion matrix: plot_confusion_matrix()
defines a function to get the class code by label: get_class_code()
defines a function to plot a tree inline: tree_to_code()
defines a function to extract all the structural arrays of a tree: get_tree_structure()
defines a function to extract a metrics dictionary from a random forest: explore_forest()
defines a function to pass batches of data to explore_forest(), split by correct/incorrect prediction: batch_analyse_model()
defines function to plot the mean path lengths from an object returned by explore_forest(): plot_mean_path_lengths()
defines a function to map the path of an instance down a tree: tree_path()
defines a function to map the path of an instance down a tree ensemble: forest_path()
defines a function to find the majority predicted class from object returned by forest_path(): major_class_from_forest_paths()
defines a function to convert a tree into a function: tree_to_code()
defines a function to get list of all the paths for one instance out of a forest_paths object: get_paths()
defines a function to get basic labels back from one hot encoded label names: decode onehot
defines a function to get basic labels back from one hot encoded label names for all paths in a get_paths() object: decode onehot
defines a function with an apriori based algorithm for finding frequent patterns from a list of paths: apriori()
defines a function to sort a frequent pattern list returned by apriori()
defines a class to accumulate freqent patterns into a rule set rule_accumulator
defines a function to apply an accumulated rule to a data set: apply_rule()
''')
