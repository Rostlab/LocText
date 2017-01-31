"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.model_selection.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from util import my_cv_generator

print(__doc__)

annotator, X, y = get_model_and_data()

search_space = [
    {
        'kernel': ['rbf'],
        'class_weight': [None, 'balanced'],
        'C': [2**log2 for log2 in list(range(-7, 15, 1))],
        'gamma': [2**log2 for log2 in list(range(3, -15, -2))],
    },
    {
        'kernel': ['linear'],
        'class_weight': [None, 'balanced'],
        'C': [2**log2 for log2 in list(range(-7, 15, 1))],
    }
]

scores = [
    # 'accuracy',
    'f1_macro',
    'precision_macro',
    'recall_macro'
]

for score in scores:
    print()
    print()
    print("# Tuning hyper-parameters for *** {} ***".format(score))
    print()

    grid = GridSearchCV(
        estimator=SVC(C=1, verbose=False),  # TODO C=1 ??
        param_grid=search_space,
        verbose=True,
        cv=my_cv_generator(len(y)),
        scoring=score,
        refit=False,
        iid=False,
    )

    grid.fit(X, y)

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
