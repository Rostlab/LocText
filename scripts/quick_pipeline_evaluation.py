import sys

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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import RandomizedLogisticRegression

sentence_distance = int(sys.argv[1])
use_pred = sys.argv[2].lower() == "true"

print(sentence_distance, use_pred)

# ----------------------------------------------------------------------------------------------------

annotator, X, y, groups = get_model_and_data(sentence_distance, use_pred)
X = X.toarray()

print("Shape X, before: ", X.shape)

feature_selections = [
    ("LinearSVC_C=1", SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False, random_state=2727, tol=1e-5))),
    ("LinearSVC_C=0.5", SelectFromModel(LinearSVC(C=0.5, penalty="l1", dual=False, random_state=2727, tol=1e-5))),
    ("LinearSVC_C=0.25", SelectFromModel(LinearSVC(C=0.25, penalty="l1", dual=False, random_state=2727, tol=1e-5))),

    ("RandomizedLogisticRegression_C=1", SelectFromModel(RandomizedLogisticRegression(C=1))),
    ("RandomizedLogisticRegression_C=0.5", SelectFromModel(RandomizedLogisticRegression(C=0.5))),

    # ("PCA_2", PCA(2)),
    # ("PCA_10", PCA(2)),
    # ("PCA_100", PCA(2)),
    # ("PCA_400", PCA(2)),
    #
    # ("TruncatedSVD_2", TruncatedSVD(2)),
    # ("TruncatedSVD_10", TruncatedSVD(2)),
    # ("TruncatedSVD_100", TruncatedSVD(2)),
    # ("TruncatedSVD_400", TruncatedSVD(2)),
    # ("LogisticRegression", SelectFromModel(LogisticRegression(penalty="l1"))),
    # ("RandomForestClassifier_20", SelectFromModel(RandomForestClassifier(n_estimators=20, max_depth=3))),
    # ("RandomForestClassifier_100", SelectFromModel(RandomForestClassifier(n_estimators=100, max_depth=None))),
]

estimators = [
    ("SVC_linear", SVC(kernel='linear')),
    # ("SVC_rbf", SVC(kernel='rbf')),
    # ("RandomForestClassifier_20", RandomForestClassifier(n_estimators=20, max_depth=3)),
    # ("RandomForestClassifier_100", RandomForestClassifier(n_estimators=100, max_depth=5)),
]

for fsel_name, feature_selection in feature_selections:

    X_new = feature_selection.fit_transform(X, y)
    print()
    print()
    print()
    print(fsel_name, " --- ", X_new.shape)
    print()

    selected_feature_keys = feature_selection.get_support(indices=True)
    fsel_names, _ = print_selected_features(
        selected_feature_keys,
        annotator.pipeline.feature_set,
        file_prefix=("_".join([str(sentence_distance), str(use_pred), fsel_name]))
    )
    print(fsel_names)

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
