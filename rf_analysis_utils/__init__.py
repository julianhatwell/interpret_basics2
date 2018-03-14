from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import graphviz
import pydotplus
from IPython.display import Image
# import itertools
# import timeit
import math
# import operator

from operator import itemgetter
from collections import deque, defaultdict
from itertools import combinations, product
from copy import deepcopy

# import lime
# import lime.lime_tabular as limtab
# from treeinterpreter import treeinterpreter as ti, utils

import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
# from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn.pipeline import make_pipeline

# from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import itemfreq, sem, entropy
