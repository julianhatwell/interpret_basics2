import urllib
# import sys
# import os
import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
# import pickle
# from itertools import chain

from forest_walker import structures as sts
# from pandas import read_csv

# credit
# data from source
def credit_data_from_source():
    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'

    credit_bytes = urllib.request.urlopen(target_url)
    credit = pd.read_csv(credit_bytes,
                         header=None,
                         delimiter=',',
                         index_col=False,
                         names=var_names,
                         na_values = '?')

    # re-code rating class variable
    A16 = pd.Series(['plus'] * credit.shape[0])
    A16.loc[credit.A16.values == '-'] = 'minus'
    credit.A16 = A16

    # deal with some missing data
    for v, t in zip(var_names, vars_types):
        if t == 'nominal':
            credit[v] = credit[v].fillna('u')
        else:
            credit[v] = credit[v].fillna(credit[v].mean())

    return(credit)

def credit_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_walker' + sts.path_sep + 'datafiles' + sts.path_sep + 'credit.csv.gz',
                    compression='gzip'),
    class_col = 'A16',
    var_names = ['A1'
                , 'A2'
                , 'A3'
                , 'A4'
                , 'A5'
                , 'A6'
                , 'A7'
                , 'A8'
                , 'A9'
                , 'A10'
                , 'A11'
                , 'A12'
                , 'A13'
                , 'A14'
                , 'A15'
                , 'A16'],
    var_types = ['nominal'
                , 'continuous'
                , 'continuous'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'continuous'
                , 'nominal'
                , 'nominal'
                , 'continuous'
                , 'nominal'
                , 'nominal'
                , 'continuous'
                , 'continuous'
                , 'nominal'],
    pickle_dir = 'credit_pickles',
    random_state=123,
    spiel = '''
    Data Set Information:

    This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

    This dataset is interesting because there is a good mix of attributes -- continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.

    Attribute Information:

    A1:	b, a.
    A2:	continuous.
    A3:	continuous.
    A4:	u, y, l, t.
    A5:	g, p, gg.
    A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    A7:	v, h, bb, j, n, z, dd, ff, o.
    A8:	continuous.
    A9:	t, f.
    A10:	t, f.
    A11:	continuous.
    A12:	t, f.
    A13:	g, p, s.
    A14:	continuous.
    A15:	continuous.
    A16: +,- (class attribute)
    ''')

    return(data_cont)
