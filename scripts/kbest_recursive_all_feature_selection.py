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
from sklearn.preprocessing import FunctionTransformer

print(__doc__)

SCORING_FUNCS = [mutual_info_classif]
SCORING_NAMES = ['f1_macro']

annotator, X, y = get_model_and_data()
X_transformed = X  # Keep for historical reasons; we experimented what's fastest csr or csc -- http://stackoverflow.com/questions/41998147/what-is-row-slicing-vs-what-is-column-slicing

num_instances, num_features = X.shape

for scoring_name in SCORING_NAMES:
    for scoring_func in SCORING_FUNCS:

        kbest = SelectKBest(scoring_func, k="all")
        kbest.fit(X, y)
        sorted_kbest_feature_keys = get_sorted_kbest_feature_keys(kbest)

        scores = []

        start = time.time()
        for num_seletected_kbest_features in range(1, num_features + 1):

            selected_feature_keys = sorted_kbest_feature_keys[:num_seletected_kbest_features]
            my_transformer = FunctionTransformer(select_features_transformer_function, accept_sparse=True, kw_args={"selected_feature_keys": selected_feature_keys})

            svc = SVC(kernel='linear', C=1, verbose=False)  # TODO C=1 linear / rbf ??
            estimator = make_pipeline(my_transformer, svc)

            cv_scores = cross_val_score(estimator, X_transformed, y, scoring=scoring_name, cv=my_cv_generator(num_instances), verbose=True, n_jobs=-1)
            scores.append(cv_scores.mean())

        end = time.time()
        print("\n\n{} : {} -- TIME for feature selection : {}".format(scoring_name, scoring_func, (end - start)))

        assert(len(scores) == num_features)

        print()
        # print(print_selected_features(selected_feat_keys, annotator.pipeline.feature_set, file_prefix="rfe"))
        print()
        print("Max performance for {}: {}".format(scoring_name, max(scores)))
        print()

        plot_recursive_features(scoring_name, scores)