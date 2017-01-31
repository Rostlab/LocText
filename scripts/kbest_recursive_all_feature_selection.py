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
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from util import *
from loctext.util import *
import time
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

print(__doc__)

SCORING_FUNCS = ["stub"]
SCORING_NAMES = ['f1_macro']

annotator, X, y = get_model_and_data()

num_instances, num_features = X.shape

search_space = [
    {
        'k': list(range(1, num_features + 1)),
    },
]

for scoring_name in SCORING_NAMES:
    for scoring_func in SCORING_FUNCS:

        kbest = SelectKBest(scoring_func, k="all")
        kbest.fit(X, y)
        selected_feat_keys = get_kbest_feature_keys()

        scores = []

        for num_seletected_kbest_features in range(1, num_features + 1):

            allowed_feat_keys = selected_feat_keys[:num_seletected_kbest_features]

            estimator = make_pipeline(MY_TRANSFORMATION, SVC(kernel='linear', C=1, verbose=False))  # TODO C=1 linear / rbf ??

            cv_scores = cross_val_score(estimator, X, y, scoring=scoring_name, cv=my_cv_generator(num_instances))
            scores.append(cv_scores.append(cv_scores.mean()))

        assert(len(scores) == num_features)

        print()
        # print(print_selected_features(selected_feat_keys, annotator.pipeline.feature_set, file_prefix="rfe"))
        print()
        print("Max performance for {}: {}".format(scoring_name, max(scores)))
        print()

        plot_recursive_features(scoring_name, scores)
