"""
============================================================
Feature Selection
============================================================

...

"""

from __future__ import print_function

import random

from scipy import sparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset

from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from nalaf import is_debug_mode, is_verbose_mode

print(__doc__)

is_debug_mode = is_verbose_mode = True

corpus = read_corpus("LocText")
locTextModel = LocTextSSmodelRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID)
locTextModel.pipeline.execute(corpus, train=True)
X, y = locTextModel.model.write_vector_instances(corpus, locTextModel.pipeline.feature_set)

print("Step 1, # features: ", X.shape[1])

# # See: http://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
p = 0.99
selector = VarianceThreshold(threshold=(p * (1 - p)))
X = selector.fit_transform(X)

print("Step 2, # features", X.shape[1], " -- variances ", selector.variances_)
