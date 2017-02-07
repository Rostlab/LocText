"""
recursive `grid` and `all` should yield the same results (tested for few features) -- `all` should be faster (not tested)
"""

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.learning.annotators import LocTextDXModelRelationExtractor
from util import *
from loctext.util import *
import time

print(__doc__)

SCORING_FUNCS = [mutual_info_classif]
SCORING_NAMES = ['f1']

annotator, X, y, groups = get_model_and_data()

num_instances, num_features = X.shape

search_space = [
    {
        'k': list(range(1, num_features + 1)),
    },
]

for scoring_name in SCORING_NAMES:
    for scoring_func in SCORING_FUNCS:

        estimator = KBestSVC(X, y, score_func=scoring_func)

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=search_space,
            verbose=True,
            cv=my_cv_generator(groups, num_instances),
            scoring=scoring_name,
            refit=False,
            iid=False,
        )

        start = time.time()
        X_new = grid.fit(X, y)
        end = time.time()

        print("TIME for feature selection: ", (end - start))

        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)
        print()
        print("Grid scores on development set:")
        print()

        scores = grid.cv_results_['mean_test_score']
        assert(len(scores) == len(grid.cv_results_["params"]) == num_features)

        print()
        # print(print_selected_features(selected_feat_keys, annotator.pipeline.feature_set, file_prefix="rfe"))
        print()
        print("Max performance for {}: {}".format(scoring_name, max(scores)))
        print()

        plot_recursive_features(scoring_name, scores)
