# experimental script
import json
from forest_surveyor.datasets import credit_data
from forest_surveyor.structures import forest_walker, batch_getter
from forest_surveyor.routines import tune_rf, train_rf, evaluate_model, walk_paths, cor_incor_forest_survey
from forest_surveyor.plotting import log_ratio_plot
import forest_surveyor.routines as rtn

# load a data set
mydata = credit_data()

# train test split
tt = mydata.tt_split()

################ PARAMETER TUNING ###################
############# Only run when required ################
#####################################################

### params = tune_rf(tt['X_train_enc'], tt['y_train'],
###  save_path = mydata.pickle_path(),
###  random_state=mydata.random_state)

#####################################################
#####################################################
#####################################################

# load previously tuned parameters
with open(mydata.pickle_path('params.json'), 'r') as infile:
    params = json.load(infile)

# train a rf model
rf, enc_rf = train_rf(X=tt['X_train_enc'], y=tt['y_train'],
 params=params,
 encoder=tt['encoder'],
 random_state=mydata.random_state)

# only in interactive mode - evaluate and plot the confusion matrix
# evaluate_model(prediction_model=enc_rf, X=tt['X_test'], y=tt['y_test'], class_names=mydata.class_names)

# fit the forest_walker
f_walker = forest_walker(forest = rf,
 features=mydata.onehot_features,
 encoder=tt['encoder'],
 prediction_model=enc_rf)

# run the batch based forest walker
getter = batch_getter(instances=tt['X_test'], labels=tt['y_test'])
instances, labels = getter.get_next()

paths = f_walker.forest_walk(instances = instances
                            , labels = labels
                            , by_tree = False)

# walk_paths(mydata, f_walker)

# run the full forest survey
# to do - re do lighter version of survey that takes up less memory, from original forest survey function
# to do - script up other plots for completeness
# tt_correct_stats, tt_incorrect_stats = cor_incor_forest_survey(
#  f_walker = f_walker, X=tt['X_test'], y=tt['y_test'])
#
# log_ratio = log_ratio_plot(num = tt_correct_stats[0]['m_child_traversals']
#                             , num_err = tt_correct_stats[0]['se_child_traversals']
#                             , denom = tt_incorrect_stats[0]['m_child_traversals']
#                             , denom_err = tt_incorrect_stats[0]['se_child_traversals']
#                             , labels = mydata.onehot_features
# )
# log_ratio = log_ratio_plot(num = tt_correct_stats[0]['m_lower_traversals']
#                             , num_err = tt_correct_stats[0]['se_lower_traversals']
#                             , denom = tt_incorrect_stats[0]['m_lower_traversals']
#                             , denom_err = tt_incorrect_stats[0]['se_lower_traversals']
#                             , labels = mydata.onehot_features
# )
