import time
import timeit

from sklearn.ensemble import RandomForestClassifier

def mp_run_rf(X, y, g, random_state, output):

    rf = RandomForestClassifier()
    rf.set_params(oob_score = True, random_state=random_state, **g)

    print('starting new run at: ' + time.asctime(time.gmtime()) + ' with parameters: ' + str(g))
    tree_start_time = timeit.default_timer()
    rf.fit(X, y)
    tree_end_time = timeit.default_timer()
    print('ending run at: ' + time.asctime(time.gmtime()))

    g['elapsed_time'] = tree_end_time - tree_start_time
    g['score'] = rf.oob_score_

    output.put(g)
