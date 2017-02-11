from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import matplotlib.pyplot as plot
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextDXModelRelationExtractor
from util import *
from loctext.util import *
from sklearn.model_selection import cross_val_score
import time
from sklearn.pipeline import Pipeline

sentence_distance = 0
use_pred = True

annotator, X, y, groups = get_model_and_data(sentence_distance, use_pred)

print("Shape X, before: ", X.shape)

feature_selections = [
    ("LinearSVC", SelectFromModel(LinearSVC(penalty="l1", dual=False, random_state=2727, tol=1e-50))),
    # ("LogisticRegression", SelectFromModel(LogisticRegression(penalty="l1"))),
    #("RandomForestClassifier_20", SelectFromModel(RandomForestClassifier(n_estimators=20, max_depth=3))),
    #("RandomForestClassifier_100", SelectFromModel(RandomForestClassifier(n_estimators=100, max_depth=3))),
]

estimators = [
    ("SVC_linear", SVC(kernel='linear')),
    # ("SVC_rbf", SVC(kernel='rbf')),
    #("RandomForestClassifier_20", RandomForestClassifier(n_estimators=20, max_depth=3)),
    #("RandomForestClassifier_100", RandomForestClassifier(n_estimators=100, max_depth=3)),
]

for fsel_name, feature_selection in feature_selections:

    X_new = feature_selection.fit_transform(X, y)
    print()
    print()
    print()
    print(fsel_name, " --- ", X_new.shape)
    print()

    file_prefix = "_".join([str(sentence_distance), str(use_pred), fsel_name])

    selected_feature_keys = feature_selection.get_support(indices=True)
    keys, names, fig_file = \
        print_selected_features(selected_feature_keys, annotator.pipeline.feature_set, file_prefix=file_prefix)
    print(keys, names)

    print()

    for est_name, estimator in estimators:

        cv_scores = cross_val_score(
            estimator,
            X_new,
            y,
            scoring="f1",
            cv=my_cv_generator(groups, len(y)),
            verbose=True,
            n_jobs=-1
        )

        print()
        print(cv_scores.mean(), estimator)
