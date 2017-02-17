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

SCORING_NAMES = [
    'precision',
    'f1',
]

# TODO put threshold

SEARCH_SPACE = [
    {
    #     # 'feat_sel__estimator__C': [2**log2 for log2 in list(range(-3, 2, 1))],
    #     # 'feat_sel__estimator__class_weight': [None, 'balanced', {-1: 2}, {+1: 2}],
    #     # 'feat_sel__estimator__random_state': [None, 2727, 1, 5, 10],
    #     # 'feat_sel__estimator__tol': [1e-50],
    #     # 'feat_sel__estimator__max_iter': [1000, 10000],
    #     #
        'classify': [SVC()],
        'classify__kernel': ['rbf'],
        'classify__class_weight': [None, 'balanced', {-1: 2}, {+1: 2}],
        'classify__C': [2**log2 for log2 in list(range(-7, 15, 1))],
        'classify__gamma': [2**log2 for log2 in list(range(3, -15, -2))],
    },

    {
        # 'feat_sel': [SelectFromModel(estimator=LinearSVC(penalty="l1", dual=False))],
        # 'feat_sel__estimator__penalty': ['l1'],
        # 'feat_sel__estimator__dual': [False],
        # 'feat_sel__estimator__C': [1],
        # 'feat_sel__estimator__class_weight': [None, 'balanced', {-1: 2}, {+1: 2}],
        # 'feat_sel__estimator__random_state': [None, 2727, 1, 5, 10],
        # 'feat_sel__estimator__tol': [1e-50],
        # 'feat_sel__estimator__max_iter': [1000, 10000],
        #
        'classify': [SVC()],
        'classify__kernel': ['linear'],
        'classify__class_weight': [None, 'balanced', {-1: 2}, {+1: 2}],
        'classify__C': [2**log2 for log2 in list(range(-7, 15, 1))],
    },

    {
        # 'feat_sel__estimator__C': [2**log2 for log2 in list(range(-3, 2, 1))],
        # 'feat_sel__estimator__class_weight': [None, 'balanced', {-1: 2}, {+1: 2}],
        # 'feat_sel__estimator__random_state': [None, 2727, 1, 5, 10],
        # 'feat_sel__estimator__tol': [1e-50],
        # 'feat_sel__estimator__max_iter': [1000, 10000],
        #
        # see: http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
        # 'classify': [RandomForestClassifier()],
        # 'classify__max_features': [None, 'sqrt', 'log2'],
        # 'classify__max_depth': [None, 3, 5, 10, 20],
        # 'classify__bootstrap': [True, False],
        # 'classify__n_jobs': [-1],
        # 'classify__class_weight': [None, 'balanced', {-1: 2}, {+1: 2}],
    },
]

#####

annotator, X, y, groups = get_model_and_data()

pipeline = Pipeline([
    # ('feat_sel', SelectFromModel(estimator=LinearSVC(penalty="l1", dual=False))),
    ('classify', SVC(kernel="linear"))
])

feat_sel = SelectFromModel(LinearSVC(penalty="l1", dual=False, random_state=2727, tol=1e-50))
X_new = feat_sel.fit_transform(X, y)
selected_feature_keys = feat_sel.get_support(indices=True)

names, fig_file = \
    print_selected_features(selected_feature_keys, annotator.pipeline.feature_set, file_prefix='LinearSVC_random_state_2727_tol_1e-50')

print()
print(names)
print()

for scoring_name in SCORING_NAMES:
    print()
    print()
    print("# Tuning hyper-parameters for *** {} ***".format(scoring_name))
    print()

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=SEARCH_SPACE,
        verbose=True,
        cv=my_cv_generator(groups, len(y)),
        scoring=scoring_name,
        refit=True,
        iid=False,
        n_jobs=-1,
    )

    grid.fit(X_new, y)

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']

    desc_sorted_best_indices = sorted(range(len(means)), key=lambda k: (means[k] - stds[k]), reverse=True)

    for index in desc_sorted_best_indices:
        mean = means[index]
        std = stds[index]
        params = grid.cv_results_['params'][index]

        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

    print()
