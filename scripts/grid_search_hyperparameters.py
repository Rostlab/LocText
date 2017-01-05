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

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from nalaf.learning.lib.sklsvm import SklSVM
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
        'gamma': [1e-4],
        'C': [1]
    },
    # {
    #     'kernel': ['rbf'],
    #     'gamma': [1e-3, 1e-4],
    #     'C': [1, 10, 100, 1000]
    # },
    # {
    #     'kernel': ['linear'],
    #     'C': [1, 10, 100, 1000]
    # }
]

scores = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    cv = 5

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=cv, scoring=score)

    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
    # print()
