import json
import time
import timeit
import numpy as np
from forest_surveyor import p_count, p_count_corrected
import forest_surveyor.datasets as ds
from forest_surveyor.structures import rule_accumulator, forest_walker, rule_tester

from scipy.stats import chi2_contingency
from math import sqrt

# bug in sk-learn. Should be fixed in August
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

def as_chirps_explanation(f_walker, walked, batch_idx, data_container, instance_ids,
                            encoder, sample_instances, sample_labels,
                            support_paths=0.1, alpha_paths=0.5,
                            disc_path_bins=4, disc_path_eqcounts=False,
                            alpha_scores=0.5, which_trees='majority', precis_threshold=0.95):

        # process the path info for freq patt mining
        # first set the paths property on selected trees e.g. majority
        walked.set_paths(batch_idx, which_trees=which_trees)
        # discretize any numeric features
        walked.discretize_paths(data_container.var_dict,
                                bins=disc_path_bins,
                                equal_counts=disc_path_eqcounts)
        # the patterns are found but not scored and sorted yet
        walked.set_patterns(support=support_paths, alpha=alpha_paths, sort=False)

        # the patterns will be weighted by chi**2 for independence test, p-values
        weights = [] * len(walked.patterns)
        for wp in walked.patterns:
            rt = rule_tester(data_container=data_container,
                            rule=wp,
                            sample_instances=sample_instances)
            rt.sample_instances = encoder.transform(rt.sample_instances)
            idx = rt.apply_rule()
            covered = p_count_corrected(sample_labels[idx], [i for i in range(len(data_container.class_names))])['counts']
            not_covered = p_count_corrected(sample_labels[~idx], [i for i in range(len(data_container.class_names))])['counts']
            observed = np.array((covered, not_covered))
            # weights.append(sqrt(chi2_contingency(observed=observed, correction=True)[0]))

            if covered.sum() > 0 and not_covered.sum() > 0: # previous_counts.sum() == 0 is impossible
                weights.append(sqrt(chi2_contingency(observed=observed[:, np.where(observed.sum(axis=0) != 0)], correction=True)[0]))
            else:
                weights.append(max(weights))

        # now the patterns are scored and sorted
        walked.set_patterns(support=support_paths, alpha=alpha_paths, sort=True, weights=weights) # with chi2 and support sorting
        # walked.set_patterns(support=support_paths, alpha=alpha_paths, sort=True) # with only support sorting

        # run the rule accumulator with greedy precis
        ra_gprec = rule_accumulator(data_container=data_container, paths_container=walked, instance_id=instance_ids[batch_idx])
        ra_gprec.profile(encoder=encoder, sample_instances=sample_instances, sample_labels=sample_labels, greedy='precision', prediction_model=f_walker.prediction_model, precis_threshold=precis_threshold)
        ra_gprec.prune_rule()
        ra_gprec_lite = ra_gprec.lite_instance()

        # collect completed rule accumulator
        return(batch_idx, [ra_gprec_lite])
