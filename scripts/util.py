import matplotlib.pyplot as plot
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import scipy
import time

from nalaf.structures.data import Dataset
from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from loctext.util import *
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import FunctionTransformer, maxabs_scale


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


def plot_recursive_features(scoring_name, scores):
    plot.figure()
    plot.xlabel("Number of features selected")
    plot.ylabel("{}".format(scoring_name.upper()))
    plot.plot(range(1, len(scores) + 1), scores)
    plot.show()


def get_kbest_feature_keys(kbest_fitted_model):
    selected_feat_keys = []

    for fkey, _ in sorted(enumerate(kbest_fitted_model.scores_), key=lambda tuple: tuple[1], reverse=True):
        selected_feat_keys.append(fkey)

    return selected_feat_keys


class KBestSVC(BaseEstimator, ClassifierMixin):  # TODO inheriting on these ones makes any change?

    def __init__(self, X_whole, y_whole, score_func, k=None):
        self.X_whole = X_whole
        self.y_whole = y_whole

        self.score_func = score_func
        self.k = k
        self.kbest = None
        self.kbest_unfitted = True

        self.svc = SVC(kernel='linear', C=1, verbose=False)  # TODO C=1 linear / rbf ??

    def fit(self, X, y):
        if self.kbest_unfitted:
            self.kbest = SelectKBest(score_func=self.score_func, k=self.k)
            self.kbest.fit(self.X_whole, self.y_whole)
            self.kbest_unfitted = False

        X_new = self.kbest.transform(X)
        X_new = SklSVM._preprocess(X_new)
        return self.svc.fit(X_new, y)

    def predict(self, X):
        X_new = self.kbest.transform(X)
        X_new = SklSVM._preprocess(X_new)
        return self.svc.predict(X_new)


def gen_final_allowed_feature_mapping(allowed_feat_keys):
    final_allowed_feature_mapping = {}

    for allowed_feat_key in allowed_feat_keys:
        final_allowed_feature_mapping[allowed_feat_key] = len(final_allowed_feature_mapping)

    return final_allowed_feature_mapping


def select_features_transformer_function(X, **kwargs):
    allowed_feature_keys = kwargs["allowed_feature_keys"]

    # num_instances, _ = X.shape
    # num_features = len(final_allowed_feature_mapping)
    #
    # X_new = scipy.sparse.lil_matrix((num_instances, num_features), dtype=np.float64)

    # for instance_index in range(num_instances):
    #     for f_key, f_index in final_allowed_feature_mapping.items():
    #         X_new[instance_index, f_index] = X[instance_index, f_key]

    # for f_key, f_index in final_allowed_feature_mapping.items():
    #     X_new[:, f_index] = X[:, f_key]

    # X_new = X_new.tocsr()
    X_new = X[:, allowed_feature_keys]
    X_new = SklSVM._preprocess(X_new)

    return X_new


# def select_features_transformer_transformer(final_allowed_feature_mapping, accept_sparse=True):
#     transformer_fun = select_features_transformer_function(final_allowed_feature_mapping)
#     return FunctionTransformer(transformer_fun, accept_sparse=accept_sparse)
