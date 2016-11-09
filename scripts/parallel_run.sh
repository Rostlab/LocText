#!/bin/bash

id=`date +%s%N`

# Sort by F-Measure: tail -n 1 logs/*.log | sort -k 5 -t " "

for minority_class in `seq -1 2 +1`; do
  for c in None `seq 0.0005 0.0005 0.1000`; do # the None is intentional and means svm_learn default
    for majority_class_undersampling in `seq 0.05 0.05 0.50`; do
      time python loctext/learning/train.py --corpus_percentage 1.0 --minority_class $minority_class --majority_class_undersampling $majority_class_undersampling --svm_hyperparameter_c $c &> logs/"loctext_id${id}_m${minority_class}_u${majority_class_undersampling}_c${c}.log" &
    done
    wait
    for majority_class_undersampling in `seq 0.55 0.05 1`; do
      time python loctext/learning/train.py --corpus_percentage 1.0 --minority_class $minority_class --majority_class_undersampling $majority_class_undersampling --svm_hyperparameter_c $c &> logs/"loctext_id${id}_m${minority_class}_u${majority_class_undersampling}_c${c}.log" &
    done
    wait
  done
done
