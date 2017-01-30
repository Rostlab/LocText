from nalaf.structures.data import Dataset
import time

def my_cv_generator(num_instances):
    k = 5

    index_keys = list(range(0, num_instances))
    index_keys = Dataset._cv_kfold_splits_randomize_keys(index_keys)

    for fold in range(k):
        training, evaluation = Dataset._cv_kfold_split(index_keys, k, fold, validation_set=True)
        yield training, evaluation
