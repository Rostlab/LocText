import sys

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextDXModelRelationExtractor
from util import *
from loctext.util import *
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


print(__doc__)

sentence_distance = int(sys.argv[1])
use_pred = sys.argv[2].lower() == "true"

annotator, X, y, groups = get_model_and_data(sentence_distance, use_pred)

print(sentence_distance, use_pred)

# ----------------------------------------------------------------------------------------------------

SCORING_NAMES = [
    F05_SCORER,
    'f1',
]

FEATURE_SELECTIONS = [
    ("LinearSVC_C=4.0", SelectFromModel(LinearSVC(C=4.0, penalty="l1", dual=False, random_state=2727, tol=1e-5))),
    ("LinearSVC_C=2.0", SelectFromModel(LinearSVC(C=2.0, penalty="l1", dual=False, random_state=2727, tol=1e-5))),
    ("LinearSVC_C=1.0", SelectFromModel(LinearSVC(C=1.0, penalty="l1", dual=False, random_state=2727, tol=1e-5))),
]

PIPELINE = Pipeline([
    # ("feat_sel", SelectFromModel(LinearSVC(penalty="l1", dual=False, random_state=2727, tol=1e-5))),
    ('classify', SVC(kernel="linear"))
])

SEARCH_SPACE = [
    {
        'classify': [SVC()],
        'classify__kernel': ['rbf'],
        'classify__class_weight': [None, 'balanced', {-1: 1.5}, {-1: 2.0}],
        'classify__C': [2**log2 for log2 in list(range(-3, 10, 1))],
        'classify__gamma': [2**log2 for log2 in list(range(3, -15, -2))],
    },

    {
        'classify': [SVC()],
        'classify__kernel': ['linear'],
        'classify__class_weight': [None, 'balanced', {-1: 1.5}, {-1: 2.0}],
        'classify__C': [2**log2 for log2 in list(range(-3, 10, 1))],
    },

    {
        # see: http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
        'classify': [RandomForestClassifier()],
        'classify__max_features': [None, 'sqrt', 'log2'],
        'classify__max_depth': [None, 3, 5, 10, 20],
        'classify__bootstrap': [True, False],
        'classify__n_jobs': [-1],
        'classify__class_weight': [None, 'balanced', {-1: 2}, {+1: 2}],
    },
]

# ----------------------------------------------------------------------------------------------------

for fsel_name, feature_selection in FEATURE_SELECTIONS:

    X_new = feature_selection.fit_transform(X, y)

    for scoring_name in SCORING_NAMES:
        print()
        print("----------------------------------------------------------------------------------------------------")
        print()
        print("# Tuning hyper-parameters for *** {} *** with {}".format(scoring_name, fsel_name))
        print()

        grid = GridSearchCV(
            estimator=PIPELINE,
            param_grid=SEARCH_SPACE,
            verbose=True,
            cv=my_cv_generator(groups, len(y)),
            scoring=scoring_name,
            refit=False,
            iid=False,
            n_jobs=-1,
        )

        grid.fit(X_new, y)

        print()
        print("Best parameters found for *** {} *** with {}".format(scoring_name, fsel_name))
        print()
        print(grid.best_params_)
        print()
        print("Grid scores:")
        print()

        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']

        desc_sorted_best_indices = sorted(range(len(means)), key=lambda k: (means[k] - stds[k]), reverse=True)

        for index in desc_sorted_best_indices[:20]:
            mean = means[index]
            std = stds[index]
            params = grid.cv_results_['params'][index]

            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

        print()

    print()

    selected_feature_keys = feature_selection.get_support(indices=True)
    fsel_names, _ = print_selected_features(
        selected_feature_keys,
        annotator.pipeline.feature_set,
        file_prefix=("_".join([str(sentence_distance), str(use_pred), fsel_name]))
    )
    print(fsel_names)
