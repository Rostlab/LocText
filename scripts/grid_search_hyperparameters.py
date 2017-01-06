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

from __future__ import print_function

import random

from scipy import sparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset

from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor

print(__doc__)

corpus = read_corpus("LocText")
locTextModel = LocTextSSmodelRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID)
locTextModel.pipeline.execute(corpus, train=True)
X, y = SklSVM._convert_edges_to_SVC_instances(corpus, locTextModel.pipeline.feature_set)

# Set the parameters by cross-validation
tuned_parameters = [
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

# See Dataset.cv_kfold_splits
def cv_generator():
    k = 5
    num_samples = len(y)

    index_keys = list(range(0, num_samples))
    index_keys = Dataset._cv_kfold_splits_randomize_keys(index_keys)

    for fold in range(k):
        training, evaluation = Dataset._cv_kfold_split(index_keys, k, fold, validation_set=True)
        yield training, evaluation

for score in scores:
    print()
    print()
    print("# Tuning hyper-parameters for *** {} ***".format(score))
    print()

    clf = GridSearchCV(
        SVC(C=1, verbose=False),
        tuned_parameters,
        verbose=False,
        cv=cv_generator(),
        scoring=score,
        refit=False,
        iid=False,
    )

    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    desc_sorted_best_indices = sorted(range(len(means)), key=lambda k: (means[k] - stds[k]), reverse=True)

    for index in desc_sorted_best_indices:
        mean = means[index]
        std = stds[index]
        params = clf.cv_results_['params'][index]

        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print()
