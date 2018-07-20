import json
import time
import timeit
import numpy as np
from forest_surveyor import p_count, p_count_corrected
import forest_surveyor.datasets as ds
from forest_surveyor.structures import rule_accumulator, forest_walker, rule_tester

from scipy.stats import chi2_contingency
from math import sqrt

def as_chirps_explanation(rule_acc, batch_idx,
                            encoder, sample_instances, sample_labels, pred_model,
                            greedy='precision', precis_threshold=0.95):

        # run the rule accumulator with greedy precis
        rule_acc.profile(encoder=encoder,
                    sample_instances=sample_instances,
                    sample_labels=sample_labels,
                    greedy=greedy,
                    prediction_model=pred_model,
                    precis_threshold=precis_threshold)
        rule_acc.prune_rule()
        ra_lite = rule_acc.lite_instance()

        # collect completed rule accumulator
        return(batch_idx, [ra_lite])
