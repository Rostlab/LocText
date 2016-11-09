#!/bin/bash

for minority_class in `seq -1 2 +1`; do
  for majority_class_undersampling in `seq 0.05 0.05 1`; do
    for c in None `seq 0.0005 0.0005 0.1000`; do # the None is intentional and means svm_learn default
      #time echo $c 2>&1 | tee "loctext_${minority_class}_${majority_class_undersampling}_${c}.log"
      time python loctext/learning/train.py --corpus_percentage 1.0 --minority_class $minority_class --majority_class_undersampling $majority_class_undersampling --svm_hyperparameter_c $c 2>&1 | tee "loctext_m${minority_class}_u${majority_class_undersampling}_c${c}.log"
    done
  done
done
