from __future__ import print_function

import random

from scipy import sparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset

from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor

import matplotlib.pyplot as plt

print(__doc__)

corpus = read_corpus("LocText")
locTextModel = LocTextSSmodelRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID)
locTextModel.pipeline.execute(corpus, train=True)
X, y = SklSVM._convert_edges_to_SVC_instances(corpus, locTextModel.pipeline.feature_set)

def pca_plot():
    X_copy = X.toarray()
    pca_2d = PCA(n_components=2).fit_transform(X_copy)

    for instance_i in range(0, pca_2d.shape[0]):
        if y[instance_i] < 0:
            neg = plt.scatter(pca_2d[instance_i, 0], pca_2d[instance_i, 1], c='r')
        else:
            pos = plt.scatter(pca_2d[instance_i, 0], pca_2d[instance_i, 1], c='g')

    plt.legend([neg, pos], ['Negative', 'Positive'])

    plt.axis('tight')
    plt.title('PCA 2D')
    plt.show()

pca_plot()
