import numpy as np
from forest_surveyor import p_count, p_count_corrected
from collections import deque
from scipy import sparse

def tree_walk(self, tree, instances, labels = None, features = None):

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

    # structural objects from tree
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    # predictions from tree
    tree_pred = tree.predict(instances)
    if self.get_label is not None:
        tree_pred_labels = self.get_label(self.class_col, tree_pred.astype(int))
    else:
        tree_pred_labels = tree_pred
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
        pass_test = True
        if features is None:
            feature_name = None
        else:
            feature_name = features[feature[p]]
        if p == 0: # root node
            ic += 1
            feature_value = instances[ic, [feature[p]]].item(0)
            leq_threshold = feature_value <= threshold[p]
            if labels is None:
                true_class = None
            else:
                true_class = labels.values[ic]
            instance_paths[ic] = {'instance_id' : instance_ids[ic]
                                    , 'pred_class' : tree_pred[ic].astype(np.int64)
                                    , 'pred_class_label' : tree_pred_labels[ic]
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
            feature_value = instances[ic, [feature[p]]].item(0)
            leq_threshold = feature_value <= threshold[p]
            instance_paths[ic]['path']['feature_idx'].append(feature[p])
            instance_paths[ic]['path']['feature_name'].append(feature_name)
            instance_paths[ic]['path']['feature_value'].append(feature_value)
            instance_paths[ic]['path']['threshold'].append(threshold[p])
            instance_paths[ic]['path']['leq_threshold'].append(leq_threshold)

    return(instance_paths)
