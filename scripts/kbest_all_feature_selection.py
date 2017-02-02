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
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from util import *
from loctext.util import *
import time

print(__doc__)

SCORING_FUNCS = [mutual_info_classif]
SCORING_NAMES = ["stub"]

annotator, X, y = get_model_and_data()

for scoring_func in SCORING_FUNCS:
    for scoring_name in SCORING_NAMES:

        kbest = SelectKBest(scoring_func, k="all")

        start = time.time()
        kbest.fit(X, y)
        end = time.time()

        print("TIME for feature selection: ", (end - start))

        selected_feat_keys = get_sorted_kbest_feature_keys(kbest)

        print()
        print(print_selected_features(selected_feat_keys, annotator.pipeline.feature_set, file_prefix="kbest"))
