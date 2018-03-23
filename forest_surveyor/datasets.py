import urllib
import pandas as pd

from forest_surveyor import structures as sts

# adult
def adult_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'adult.csv.gz',
                        compression='gzip'),
    class_col = 'income',
    pickle_dir = 'adult_pickles',
    random_state=123,
    spiel = '''
    Data Description:
    This data was extracted from the adult bureau database found at
    http://www.adult.gov/ftp/pub/DES/www/welcome.html
    Donor: Ronny Kohavi and Barry Becker,
          Data Mining and Visualization
          Silicon Graphics.
          e-mail: ronnyk@sgi.com for questions.
    Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
    48842 instances, mix of continuous and discrete    (train=32561, test=16281)
    45222 if instances with unknown values are removed (train=30162, test=15060)
    Duplicate or conflicting instances : 6
    Class probabilities for adult.all file
    Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
    Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
    Extraction was done by Barry Becker from the 1994 adult database.  A set of
     reasonably clean records was extracted using the following conditions:
     ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    ''')
    return(data_cont)

# adult sample
def adult_samp_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'adult_samp.csv.gz',
                        compression='gzip'),
    class_col = 'income',
    pickle_dir = 'adult_pickles',
    random_state=123,
    spiel = '''
    Data Description:
    This data was extracted from the adult bureau database found at
    http://www.adult.gov/ftp/pub/DES/www/welcome.html
    Donor: Ronny Kohavi and Barry Becker,
          Data Mining and Visualization
          Silicon Graphics.
          e-mail: ronnyk@sgi.com for questions.
    Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
    48842 instances, mix of continuous and discrete    (train=32561, test=16281)
    45222 if instances with unknown values are removed (train=30162, test=15060)
    Duplicate or conflicting instances : 6
    Class probabilities for adult.all file
    Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
    Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
    Extraction was done by Barry Becker from the 1994 adult database.  A set of
     reasonably clean records was extracted using the following conditions:
     ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    ''')
    return(data_cont)

# car
def car_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'car.csv.gz',
                        compression='gzip'),
    class_col = 'acceptability',
    var_names = ['buying'
                , 'maint'
                , 'doors'
                , 'persons'
                , 'lug_boot'
                , 'safety'
                , 'acceptability'],
    pickle_dir = 'credit_pickles',
    random_state=123,
    spiel = '''
    M. Bohanec and V. Rajkovic: Knowledge acquisition and explanation for
    multi-attribute decision making. In 8th Intl Workshop on Expert
    Systems and their Applications, Avignon, France. pages 59-78, 1988.

    Within machine-learning, this dataset was used for the evaluation
    of HINT (Hierarchy INduction Tool), which was proved to be able to
    completely reconstruct the original hierarchical model. This,
    together with a comparison with C4.5, is presented in

    B. Zupan, M. Bohanec, I. Bratko, J. Demsar: Machine learning by
    function decomposition. ICML-97, Nashville, TN. 1997 (to appear)
    ''')
    return(data_cont)

# cardiotography
def cardiotography_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'cardiotography.csv.gz',
                        compression='gzip'),
    class_col = 'NSP',
    pickle_dir = 'cardiotography_pickles',
    random_state=123,
    spiel = '''
    Data Set Information:
    2126 fetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C. ...) and to a fetal state (N, S, P). Therefore the dataset can be used either for 10-class or 3-class experiments.


    Attribute Information:
    LB - FHR baseline (beats per minute)
    AC - # of accelerations per second
    FM - # of fetal movements per second
    UC - # of uterine contractions per second
    DL - # of light decelerations per second
    DS - # of severe decelerations per second
    DP - # of prolongued decelerations per second
    ASTV - percentage of time with abnormal short term variability
    MSTV - mean value of short term variability
    ALTV - percentage of time with abnormal long term variability
    MLTV - mean value of long term variability
    Width - width of FHR histogram
    Min - minimum of FHR histogram
    Max - Maximum of FHR histogram
    Nmax - # of histogram peaks
    Nzeros - # of histogram zeros
    Mode - histogram mode
    Mean - histogram mean
    Median - histogram median
    Variance - histogram variance
    Tendency - histogram tendency
    CLASS - FHR pattern class code (1 to 10)
    NSP - fetal state class code (N=normal; S=suspect; P=pathologic)
    ''')
    return(data_cont)

# credit
def credit_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'credit.csv.gz',
                    compression='gzip'),
    class_col = 'A16',
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

# german
def german_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'german.csv.gz',
                    compression='gzip'),
    class_col = 'rating',
    pickle_dir = 'german_pickles',
    random_state=123,
    spiel = '''
    Source:
    Professor Dr. Hans Hofmann
    Institut f"ur Statistik und "Okonometrie
    Universit"at Hamburg
    FB Wirtschaftswissenschaften
    Von-Melle-Park 5
    2000 Hamburg 13

    Data Set Information:
    Two datasets are provided. the original dataset, in the form provided by Prof. Hofmann, contains categorical/symbolic attributes and is in the file "german.data".
    For algorithms that need numerical attributes, Strathclyde University produced the file "german.data-numeric". This file has been edited and several indicator variables added to make it suitable for algorithms which cannot cope with categorical variables. Several attributes that are ordered categorical (such as attribute 17) have been coded as integer. This was the form used by StatLog.

    This dataset requires use of a cost matrix:
    . 1 2
    ------
    1 0 1
    -----
    2 5 0

    (1 = Good, 2 = Bad)
    The rows represent the actual classification and the columns the predicted classification.
    It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).
    ''')

    return(data_cont)

# lending
def lending_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'lending.csv.gz',
                    compression='gzip'),
    class_col = 'loan_status',
    pickle_dir = 'lending_pickles',
    random_state=123,
    spiel = '''
    Data Set Information:
    Orignates from: https://www.lendingclub.com/info/download-data.action

    See also:
    https://www.kaggle.com/wordsforthewise/lending-club

    Prepared by Nate George: https://github.com/nateGeorge/preprocess_lending_club_data
    ''')

    return(data_cont)

# nursery
def nursery_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'nursery.csv.gz',
                    compression='gzip'),
    class_col = 'decision',
    pickle_dir = 'nursery_pickles',
    random_state=123,
    spiel = '''
    Data Description:
    Nursery Database was derived from a hierarchical decision model
    originally developed to rank applications for nursery schools. It
    was used during several years in 1980's when there was excessive
    enrollment to these schools in Ljubljana, Slovenia, and the
    rejected applications frequently needed an objective
    explanation. The final decision depended on three subproblems:
    occupation of parents and child's nursery, family structure and
    financial standing, and social and health picture of the family.
    The model was developed within expert system shell for decision
    making DEX (M. Bohanec, V. Rajkovic: Expert system for decision
    making. Sistemica 1(1), pp. 145-157, 1990.).
    ''')

    return(data_cont)

# rcdv
def rcdv_data():
    data_cont = sts.data_container(
    data = pd.read_csv('forest_surveyor' + sts.path_sep + 'datafiles' + sts.path_sep + 'rcdv.csv.gz',
                    compression='gzip'),
    class_col = 'recid',
    pickle_dir = 'rcdv_pickles',
    random_state=123,
    spiel = '''
    Data Set Information:
    This is a description of the data on the file, DATA1978.
    The description was prepared by Peter Schmidt, Department of Economics, Michigan State University, East Lansing, Michigan 48824.
    The data were gathered as part of a grant from the National Institute of Justice to Peter Schmidt and Ann Witte, “Improving Predictions of Recidivism by Use of Individual Characteristics,” 84-IJ-CX-0021.
    A more complete description of the data, and of the uses to which they were put, can be found in the final report for this grant.
    Another similar dataset, contained in a file DATA1980 on a separate diskette, is also described in that report.

    The North Carolina Department of Correction furnished a data tape which was to contain information on all individuals released from a North Carolina prison during the period from July 1, 1977 through June 30, 1978.
    There were 9457 individual records on this tape. However, 130 records were deleted because of obvious defects.
    In almost all cases, the reason for deletion is that the individual’s date of release was in fact not during the time period which defined the data set.
    This left a total of 9327 individual records, and accordingly there are 9327 records on DATA1978.

    The basic sample of 9327 observations contained many observations for which one or more of the variables used in our analyses were missing.
    Specifically, 4709 observations were missing information on one or more such variables, and these 4709 observations constitute the “missing data” file.
    The other 4618 observations which contained complete information were randomly split into an “analysis file” of 1540 observations and a “validation file” of 3078 observations.

    DATA 1978 contains 9327 individual records. Each individual record contains 28 columns of data, representing the following 19 variables.

    WHITE ALCHY JUNKY SUPER MARRIED FELON WORKREL PROPTY PERSON MALE PRIORS SCHOOL RULE AGE TSERVD FOLLOW RECID TIME FILE
    1 2 3 4 5 6 7 8 9 10 11-12 13-14 15-16 17-19 20-22 23-24 25-27 28

    WHITE is a dummy (indicator) variable equal to zero if the individual is black, and equal to one otherwise. Basically, WHITE equals one for whites and zero for blacks. However, the North Carolina prison population also contains a small number of Native Americans, Hispanics, Orientals, and individuals of “other” race. They are treated as whites, by the above definition.
    ALCHY is a dummy variable equal to one if the individual’s record indicates a serious problem with alcohol, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of ALCHY is recorded as zero, but is meaningless.
    JUNKY is a dummy variable equal to one if the individual’s record indicates use of hard drugs, and equal to zero otherwise. It is important to note that for individuals in the missing data sample (FILE = 3), the value of JUNKY is recorded as zero, but is meaningless.
    SUPER is a dummy variable equal to one if the individual’s release from the sample sentence was supervised (e.g., parole), and equal to zero otherwise.
    MARRIED is a dummy variable equal to one if the individual was married at the time of release from the sample sentence, and equal to zero otherwise.
    FELON is a dummy variable equal to one if the sample conviction was for a felony, and equal to zero if it was for a misdemeanor.
    WORKREL is a dummy variable equal to one if the individual participated in the North Carolina prisoner work release program during the sample sentence, and equal to zero otherwise.
    PROPTY is a dummy variable equal to one if the sample conviction was for a crime against property, and equal to zero otherwise. A detailed listing of the crime codes which define this variable (and PERSON below) can be found in A. Witte, Work Release in North Carolina: An Evaluation of Its Post Release Effects, Chapel Hill, North Carolina: Institute for Research in Social Science.
    PERSON is a dummy variable equal to one if the sample conviction was for a crime against a person, and equal to zero otherwise. (Incidentally, note that PROPTY plus PERSON is not necessarily equal to one, because there is an additional miscellaneous category of offenses which are neither offenses against property nor offenses against a person.)
    MALE is a dummy variable equal to one if the individual is male, and equal to zero if the individual is female.
    PRIORS is the number of previous incarcerations, not including the sample sentence. The value -9 indicates that this information is missing.
    SCHOOL is the number of years of formal schooling completed. The value zero indicates that this information is missing.
    RULE is the number of prison rule violations reported during the sample sentence.
    AGE is age (in months) at time of release.
    TSERVD is the time served (in months) for the sample sentence.
    FOLLOW is the length of the followup period, in months. (The followup period is the time from relase until the North Carolina Department of Correction records were searched, in April, 1984.)
    RECID is a dummy variable equal to one if the individual returned to a North Carolina prison during the followup period, and equal to zero otherwise.
    TIME is the length of time from release from the sample sentence until return to prison in North Carolina, for individuals for whom RECID equals one. TIME is rounded to the nearest month. (In particular, note that TIME equals zero for individuals who return to prison in North Carolina within the first half month after release.) For individuals for whom RECID equals zero, the value of TIME is meaningless. For such individuals, TIME is usually recorded as zero, but it is occasionally recorded as the length of the followup period. We emphasize again that neither value is meaningful, for those individuals for whom RECID equals zero.
    FILE is a variable indicating to which data sample the individual record belongs. The value 1 indicates the analysis sample, 2 the validation sampel and 3 is missing data sample.
    ''')

    return(data_cont)
