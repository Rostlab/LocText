import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from nalaf.structures.data import Dataset
from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from loctext.util import *
import time


def my_cv_generator(num_instances):
    k = 5

    index_keys = list(range(0, num_instances))
    index_keys = Dataset._cv_kfold_splits_randomize_keys(index_keys)

    for fold in range(k):
        training, evaluation = Dataset._cv_kfold_split(index_keys, k, fold, validation_set=True)
        yield training, evaluation


def get_model_and_data():
    corpus = read_corpus("LocText")
    # TODO the specific parameters like C=1 or even `linear` are controversial -- Maybe I should I change that
    annotator = LocTextSSmodelRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, preprocess=True, kernel='linear', C=1)
    annotator.pipeline.execute(corpus, train=True)
    X, y = annotator.model.write_vector_instances(corpus, annotator.pipeline.feature_set)

    return (annotator, X, y)
