print(__doc__)

import matplotlib.pyplot as plot
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from util import *
from loctext.util import *
import time

annotator, X, y, groups = get_model_and_data()

num_instances = len(y)

SCORING_NAMES = [
    'f1'
]

for scoring_name in SCORING_NAMES:

    rfecv = RFECV(
        verbose=1,
        n_jobs=-1,
        estimator=annotator.model.model,
        step=1,
        cv=my_cv_generator(groups, num_instances),
        scoring=scoring_name
    )

    start = time.time()
    rfecv.fit(X, y)
    end = time.time()

    print("rfe", "Time for feature selection: ", (end - start))
    print("rfe", "Optimal number of features : {}".format(rfecv.n_features_))

    selected_feat_keys = []

    for index, value in enumerate(rfecv.support_):
        if value:
            selected_feat_keys.append(index)

    print()
    print(print_selected_features(selected_feat_keys, annotator.pipeline.feature_set, file_prefix="rfe"))
    print()
    print("Max performance for {}: {}".format(scoring_name, rfecv.grid_scores_[rfecv.n_features_ - 1]))
    print()

    plot_recursive_features(scoring_name, rfecv.grid_scores_)
