"""
===================================================
Recursive feature elimination with cross-validation
===================================================

A recursive feature elimination example with automatic tuning of the
number of features selected with cross-validation.
"""
print(__doc__)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from util import my_cv_generator

corpus = read_corpus("LocText")
locTextModel = LocTextSSmodelRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, preprocess=True, kernel='linear', C=1)
locTextModel.pipeline.execute(corpus, train=True)
X, y = locTextModel.model.write_vector_instances(corpus, locTextModel.pipeline.feature_set)

rfecv = RFECV(
    verbose=1,
    n_jobs=-1,
    estimator=locTextModel.model.model,
    step=1,
    cv=my_cv_generator(len(y)),
    scoring='f1_macro'
)

rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
print("Optimal support", rfecv.support_)
print("Optimal ranking", rfecv.ranking_)
print("Optimal grid scores", rfecv.grid_scores_)


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
