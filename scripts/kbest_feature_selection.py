"""
===================================================
KBest feature Selection
===================================================
"""
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from util import my_cv_generator
import time

print(__doc__)

corpus = read_corpus("LocText")
locTextModel = LocTextSSmodelRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, preprocess=True, kernel='linear', C=1)
locTextModel.pipeline.execute(corpus, train=True)
X, y = locTextModel.model.write_vector_instances(corpus, locTextModel.pipeline.feature_set)

kbest = SelectKBest(mutual_info_classif, k='all')

start = time.time()
X_new = kbest.fit_transform(X, y)
end = time.time()

print("TIME for feature selection: ", (end - start))

print("Optimal number of features: ", X_new.shape[1])

selected = []
nonselected = []

for fkey, _ in sorted(enumerate(kbest.scores_), key=lambda tuple: tuple[1], reverse=True):
    selected.append(fkey)

print(selected)
