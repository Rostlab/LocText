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
import recursive_feature_elimination_feature_selection

print(__doc__)

SCORING_FUNCS = [mutual_info_classif]
SCORING_NAMES = ['f1']

annotator, X, y, groups = get_model_and_data()

# See also: http://stackoverflow.com/questions/41998147/what-is-row-slicing-vs-what-is-column-slicing
X_transformed = X

num_instances, num_features = X.shape

MAX_NUM_FEATURES = 38
EXTRA_FEATURES_PADDING = 0

for scoring_name in SCORING_NAMES:
    for scoring_func in SCORING_FUNCS:

        kbest = SelectKBest(scoring_func, k="all")
        kbest.fit(X, y)
        sorted_kbest_feature_keys = get_sorted_kbest_feature_keys(kbest)

        scores = []

        start = time.time()
        for num_seletected_kbest_features in range(1, min(MAX_NUM_FEATURES, num_features) + 1):

            selected_feature_keys = sorted_kbest_feature_keys[:num_seletected_kbest_features]
            my_transformer = select_features_transformer(selected_feature_keys)

            estimator = make_pipeline(my_transformer, annotator.model.model)

            cv_scores = cross_val_score(
                estimator,
                X_transformed,
                y,
                scoring=scoring_name,
                cv=my_cv_generator(groups, num_instances),
                verbose=True,
                n_jobs=-1
            )

            scores.append(cv_scores.mean())

        end = time.time()
        print("kbest_r_a", "\n\n{} : {} -- Time for feature selection : {}".format(scoring_name, scoring_func, (end - start)))

        assert(len(scores) == min(MAX_NUM_FEATURES, num_features))

        best_index, best_scoring = max(enumerate(scores), key=lambda x: x[1])
        best_num_selected_features = best_index + 1  # the first index starts in 0, this means 1 feature

        selected_feature_keys = sorted_kbest_feature_keys[:(best_num_selected_features + EXTRA_FEATURES_PADDING)]

        print()
        print()
        print("kbest_r_a", "Max performance for {}, #features={}: {}".format(scoring_name, best_num_selected_features, best_scoring))
        print()
        print()

        keys, names, fig_file = \
            print_selected_features(selected_feature_keys, annotator.pipeline.feature_set, file_prefix="kbest_r_a")

        print()
        print("\n".join([keys, names, fig_file]))
        print()

        plot_recursive_features(scoring_name, scores, save_to=fig_file, show=False)

        print()
        print()

        recursive_feature_elimination_feature_selection.call(annotator, X, y, groups, selected_feature_keys)
