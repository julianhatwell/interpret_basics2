import urllib
import pandas as pd
from datetime import datetime
import julian

# accident from Source
if True:
    '''
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import julian
    jul_conv = lambda x : 0 if x[0] == 'nan' or x[1] == 'nan' else julian.to_jd(datetime.strptime(x[0] + ' ' + x[1], '%d/%m/%Y %M:%S'))

    pickle_dir = 'accident_pickles'

    # helper function for data frame str / summary
    def rstr(df):
        return df.shape, df.apply(lambda x: [x.unique()])

    # random seed for train test split and sampling
    random_state = 123

    vtypes = {'Accident_Index' : object, 'Location_Easting_OSGR' : np.float64, 'Location_Northing_OSGR' : np.float64,
           'Longitude' : np.float64, 'Latitude' : np.float64, 'Police_Force' : np.uint8, 'Accident_Severity' : np.uint8,
           'Number_of_Vehicles' : np.uint8, 'Number_of_Casualties' : np.uint8, 'Date' : object, 'Day_of_Week' : np.uint8,
           'Time' : object, 'Local_Authority_(District)' : np.uint16, 'Local_Authority_(Highway)' : object,
           '1st_Road_Class' : np.uint8, '1st_Road_Number' : np.uint16, 'Road_Type' : np.float16, 'Speed_limit' : np.float16,
           'Junction_Detail' : np.float16, 'Junction_Control' : np.float16, '2nd_Road_Class' : np.float16,
           '2nd_Road_Number' : np.float16, 'Pedestrian_Crossing-Human_Control' : np.float16,
           'Pedestrian_Crossing-Physical_Facilities' : np.float16, 'Light_Conditions' : np.float16,
           'Weather_Conditions' : np.float16, 'Road_Surface_Conditions' : np.float16,
           'Special_Conditions_at_Site' : np.float16, 'Carriageway_Hazards' : np.float16,
           'Urban_or_Rural_Area' : np.uint8, 'Did_Police_Officer_Attend_Scene_of_Accident' : np.uint8,
           'LSOA_of_Accident_Location' : object}

    accident = pd.read_csv('data_source_files\\Accidents.csv', dtype=vtypes, low_memory=False)

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


# car form source
if True:
    '''
    var_names = ['buying'
                , 'maint'
                , 'doors'
                , 'persons'
                , 'lug_boot'
                , 'safety'
                , 'acceptability']

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'

    car_bytes = urllib.request.urlopen(target_url)
    car = pd.read_csv(car_bytes, header=None, names=var_names)
    # recode to a 2 class subproblems
    car.acceptability.loc[car.acceptability != 'unacc'] = 'acc'

    car.to_csv(pickle_path('car.csv.gz'), index=False, compression='gzip')
    '''

# credit from source
if True:
    '''
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
                , 'A16']

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

# adult
if True:
    '''
    var_names = ['age'
               , 'workclass'
               , 'lfnlwgt'
               , 'education'
               , 'educationnum'
               , 'maritalstatus'
               , 'occupation'
               , 'relationship'
               , 'race'
               , 'sex'
               , 'lcapitalgain'
               , 'lcapitalloss'
               , 'hoursperweek'
               , 'nativecountry'
               , 'income']

    vars_types = ['continuous'
               , 'nominal'
               , 'continuous'
               , 'nominal'
               , 'continuous'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'continuous'
               , 'continuous'
               , 'continuous'
               , 'nominal'
               , 'nominal']

    class_col = 'income'
    features = [vn for vn in var_names if vn != class_col]

    url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    s=requests.get(url).content
    adult_train = pd.read_csv(io.StringIO(s.decode('utf-8')), names=var_names)

    url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    s=requests.get(url).content
    adult_test = pd.read_csv(io.StringIO(s.decode('utf-8')), names=var_names, skiprows=1)

    # combine the two datasets and split them later with standard code
    frames = [adult_train, adult_test]
    adult = pd.concat(frames)

    # some tidying required
    adult.income = adult.income.str.replace('.', '')
    for f, t in zip(var_names, vars_types):
        if t == 'continuous':
            adult[f] = adult[f].astype('int32')
        else:
            adult[f] = adult[f].str.replace(' ', '')
    qm_to_unk = lambda w: 'Unknown' if w == '?' else w
    tt_fix = lambda w: 'Trinidad and Tobago' if w == 'Trinadad&Tobago' else w
    adult['workclass'] = adult.workclass.apply(qm_to_unk)
    adult['nativecountry'] = adult.nativecountry.apply(qm_to_unk)
    adult['nativecountry'] = adult.nativecountry.apply(tt_fix)

    lending['lcaptialgain'] = np.log(lending['lcaptialgain'] + abs(lending['lcaptialgain'].min()) + 1)
    lending['lcaptialloss'] = np.log(lending['lcaptialloss'] + abs(lending['lcaptialloss'].min()) + 1)
    lending['lfnlwgt'] = np.log(lending['lfnlwgt'] + abs(lending['lfnlwgt'].min()) + 1)

    # create a small set that is easier to play with on a laptop
    adult_samp = adult.sample(frac=0.25, random_state=seed).reset_index()
    adult_samp.drop(labels='index', axis=1, inplace=True)

    adult.to_csv(pickle_path('adult.csv.gz'), index=False, compression='gzip')
    adult_samp.to_csv(pickle_path('adult_samp.csv.gz'), index=False, compression='gzip')
    '''

# cardiotography
if True:
    '''
    var_names = ['LB'
                , 'AC'
                , 'FM'
                , 'UC'
                , 'DL'
                , 'DS'
                , 'DP'
                , 'ASTV'
                , 'MSTV'
                , 'ALTV'
                , 'MLTV'
                , 'Width'
                , 'Min'
                , 'Max'
                , 'Nmax'
                , 'Nzeros'
                , 'Mode'
                , 'Mean'
                , 'Median'
                , 'Variance'
                , 'Tendency'
                , 'CLASS'
                , 'NSP']

    cardiotography = pd.read_excel(pickle_path('CTG.xlsx')
                                    , header=None
                                    , names=var_names)

    # re-code NSP and delete class variable
    NSP = pd.Series(['N'] * cardiotography.shape[0])
    NSP.loc[cardiotography.NSP.values == 2] = 'S'
    NSP.loc[cardiotography.NSP.values == 3] = 'P'
    cardiotography.NSP = NSP

    to_be_del = ['CLASS']
    for tbd in to_be_del:
        del cardiotography[tbd]
        del vars_types[np.where(np.array(var_names) == tbd)[0][0]]
        del var_names[np.where(np.array(var_names) == tbd)[0][0]]
        del features[np.where(np.array(features) == tbd)[0][0]]

    cardiotography.to_csv(pickle_path('cardiotography.csv.gz'), index=False, compression='gzip')
    '''

# german
if True:
    '''
    var_names = ['chk'
                , 'dur'
                , 'crhis'
                , 'pps'
                , 'amt'
                , 'svng'
                , 'emp'
                , 'rate'
                , 'pers'
                , 'debt'
                , 'res'
                , 'prop'
                , 'age'
                , 'plans'
                , 'hous'
                , 'creds'
                , 'job'
                , 'deps'
                , 'tel'
                , 'foreign'
                , 'rating']

    target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

    german_bytes = urllib.request.urlopen(target_url)
    german = pd.read_csv(german_bytes,
                         header=None,
                         delimiter=' ',
                         index_col=False,
                         names=var_names)

    # re-code rating class variable
    rating = pd.Series(['good'] * german.count()[0])
    rating.loc[german.rating == 2] = 'bad'
    german.rating = rating

    # kill continuous vars for now
    # to_be_del = ['dur', 'amt', 'rate', 'res', 'age', 'creds', 'deps']
    #for tbd in to_be_del:
    #    del german[tbd]
    #    del vars_types[np.where(np.array(var_names) == tbd)[0][0]]
    #    del var_names[np.where(np.array(var_names) == tbd)[0][0]]
    #    del features[np.where(np.array(features) == tbd)[0][0]]

    german.to_csv(pickle_path('german.csv.gz'), index=False, compression='gzip')
    '''

# lending
if True:
    '''
    lending = pd.read_csv(pickle_path('accepted_2007_to_2017Q3.csv.gz'), compression='gzip', low_memory=True)
    # low_memory=False prevents mixed data types in the DataFrame

    # Just looking at loans that met the policy and were either fully paid or charged off (finally defaulted)
    lending = lending.loc[lending['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    # data set is wide. What can be done to reduce it?

    # drop cols with only one distinct value
    drop_list = []
    for col in lending.columns:
        if lending[col].nunique() == 1:
            drop_list.append(col)

    lending.drop(labels=drop_list, axis=1, inplace=True)

    # drop cols with excessively high missing amounts
    drop_list = []
    for col in lending.columns:
        if lending[col].notnull().sum() / lending.shape[0] < 0.5:
            drop_list.append(col)

    lending.drop(labels=drop_list, axis=1, inplace=True)

    # more noisy columns
    lending.drop(labels=['id', 'title', 'emp_title', 'application_type', 'acc_now_delinq', 'num_tl_120dpd_2m', 'num_tl_30dpd'], axis=1, inplace=True) # title is duplicated in purpose

    # convert dates to integers
    for date_col in ['issue_d', 'last_credit_pull_d', 'earliest_cr_line', 'last_pymnt_d']:
        lending[date_col] = lending[date_col].map(jul_conv)

    # highly correlated with default
    lending.drop(labels=['collection_recovery_fee', 'debt_settlement_flag', 'recoveries'], axis=1, inplace=True)

    # convert 'term' to int
    lending['term'] = lending['term'].apply(lambda s:np.float(s[1:3])) # There's an extra space in the data for some reason

    # convert sub-grade to float and remove grade
    grade_dict = {'A':0.0, 'B':1.0, 'C':2.0, 'D':3.0, 'E':4.0, 'F':5.0, 'G':6.0}
    def grade_to_float(s):
        return 5 * grade_dict[s[0]] + np.float(s[1]) - 1
    lending['sub_grade'] = lending['sub_grade'].apply(lambda s: grade_to_float(s))
    lending.drop(labels='grade', axis=1, inplace=True)

    # convert emp_length - assume missing and < 0 is no job or only very recent started job
    # emp length is only significant for values of 0 or not 0
    def emp_conv(e):
        if pd.isnull(e):
            return 'U'
        elif e[0] == '<':
            return 'U'
        else:
            return 'E'
    lending['emp'] = lending['emp_length'].apply(emp_conv)
    lending.drop(labels='emp_length', axis=1, inplace=True)

    # tidy up some very minor class codes in home ownership
    lending['home_ownership'] = lending['home_ownership'].apply(lambda h: 'OTHER' if h in ['ANY', 'NONE'] else h)

    # there is a number of rows that have missing data for many variables in a block pattern -
    # these are probably useless because missingness goes across so many variables
    # it might be possible to save them to a different set and create a separate model on them

    # another approach is to fill them with an arbitrary data point (means, zeros, whatever)
    # and add a new feature that is binary for whether this row had missing data
    # this will give the model something to adjust/correlate/associate with if these rows turn out to add noise

    # 'avg_cur_bal is a template for block missingness
    # will introduce a missing indicator column based on this
    # then fillna with zeros and finally filter out some unsalvageable really rows
    lending['block_missingness'] = lending['avg_cur_bal'].isnull() * 1.0

    # this one feature has just a tiny number of zero (invalid date) and NaNs.
    # will put these in into the same state and deal with them together
    lending['last_credit_pull_d'] = lending['last_credit_pull_d'].apply(lambda x: np.nan if x == 0 else x)
    lending['last_credit_pull_d'] = lending.last_credit_pull_d.fillna(lending.last_credit_pull_d.mean())

    lending = lending.fillna(0)
    # rows where last_pymnt_d is zero are just a mess, get them outa here
    lending = lending[lending.last_pymnt_d != 0]

    # no need for an upper and lower fico, they are perfectly correlated
    fic = ['fico_range_low', 'fico_range_high']
    lastfic = ['last_fico_range_low', 'last_fico_range_high']
    lending['fico'] = lending[fic].mean(axis=1)
    lending['last_fico'] = lending[lastfic].mean(axis=1)
    lending.drop(labels=fic + lastfic, axis=1, inplace=True)

    # slightly more informative coding of these vars that are mostly correlated with loan amnt and/or high skew
    lending['non_funded_score'] = np.log(lending['loan_amnt'] + 1 - lending['funded_amnt'])
    lending['non_funded_inv_score'] = np.log(lending['loan_amnt'] + 1 - lending['funded_amnt_inv'])
    lending['adj_log_dti'] = np.log(lending['dti'] + abs(lending['dti'].min()) + 1)
    lending['log_inc'] = np.log(lending['annual_inc'] + abs(lending['annual_inc'].min()) + 1)
    lending.drop(['funded_amnt', 'funded_amnt_inv', 'dti', 'annual_inc'], axis=1, inplace=True)

    # and rearrange so class_col is at the end
    class_col = 'loan_status'
    pos = np.where(lending.columns == class_col)[0][0]
    var_names = list(lending.columns[:pos]) + list(lending.columns[pos + 1:]) + list(lending.columns[pos:pos + 1])
    lending = lending[var_names]

    # create a small set that is easier to play with on a laptop
    lend_samp = lending.sample(frac=0.1, random_state=seed).reset_index()
    lend_samp.drop(labels='index', axis=1, inplace=True)

    # full set
    lending.to_csv(pickle_path('lending.csv.gz'), index=False, compression='gzip')
    lend_samp.to_csv(pickle_path('lend_samp.csv.gz'), index=False, compression='gzip')
    '''

# nursery
if True:
    '''
    var_names = ['parents'
               , 'has_nurs'
               , 'form'
               , 'children'
               , 'housing'
               , 'finance'
               , 'social'
               , 'health'
               , 'decision']

    vars_types = ['nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal'
               , 'nominal']

    nursery = pd.read_csv(pickle_path('nursery.csv')
                          , names=var_names)

    # filter one row where class == 2
    nursery = nursery[nursery.decision != 'recommend']
    # reset the pandas index
    nursery.index = range(len(nursery))

    nursery.to_csv(pickle_path('nursery.csv.gz'), index=False, compression='gzip')
    '''

# rcdv
if True:
    '''
    rcdv = pd.read_excel(pickle_path('rcdv.xlsx')
                                    , sheet_name='1978'
                                    , header=0)
    rcdv = rcdv.append(pd.read_excel(pickle_path('rcdv.xlsx')
                                    , sheet_name='1980'
                                    , header=0))

    # merging two sheets brought in row numbers
    rcdv.reset_index()

    # remove cols we don't want. rename file to miss as it is needed to indicate where were missing values
    rcdv.drop(labels='Column1', axis=1, inplace=True)
    rcdv.columns = ['miss' if vn == 'file' else vn for vn in rcdv.columns]

    var_names=list(rcdv)[:16] + list(rcdv)[17:] + list(rcdv)[16:17] # put recid to the end
    rcdv = rcdv[var_names]

    vars_types = ['nominal'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'nominal'
                , 'continuous'
                , 'continuous'
                , 'continuous'
                , 'continuous'
                , 'continuous'
                , 'continuous'
                , 'nominal'
                , 'continuous'
                , 'nominal'
                , 'nominal']

    class_col = 'recid'
    features = [vn for vn in var_names if vn != class_col]

    # recode priors, all that were set to -9 were missing, and it is logged in the file variable (3 = missing data indicator)
    rcdv['priors'] = rcdv['priors'].apply(lambda x: 0 if x == -9 else x)
    rcdv['miss'] = rcdv['miss'].apply(lambda x: 1 if x == 3 else 0)

    # remove cols we don't want. Time is only useful in survival analysis. Correlates exactly with recid.
    to_be_del = ['time']
    for tbd in to_be_del:
        del rcdv[tbd]
        del vars_types[np.where(np.array(var_names) == tbd)[0][0]]
        del var_names[np.where(np.array(var_names) == tbd)[0][0]]
        del features[np.where(np.array(features) == tbd)[0][0]]

    # save it out
    rcdv.to_csv(pickle_path('rcdv.csv.gz'), index=False, compression='gzip')
    '''
