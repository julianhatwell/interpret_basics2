# credit from source
if True:
    '''
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
    '''
