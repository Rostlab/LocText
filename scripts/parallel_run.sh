#!/bin/bash

id=`date +%s%N`
logsdir="$logsdir/$id"
mkdir -p $logsdir

# Sort by F-Measure:
#  * oldIFS=$IFS; IFS=$'\n'; time for f in `tail -n 1 $logsdir/*.log | grep -oP "(?<=f_measure=).*" | sort -b -k 1,1 -t " " -h | tail -n 1000`; do grep -o "f_measure=$f" $logsdir/*.log; done | tee $logsdir/loctext_best_1000_runs_id${id}.log; IFS=$oldIFS

for minority_class in `seq -1 2 +1`; do
  for c in None `seq 0.0005 0.0005 0.1000`; do # the None is intentional and means svm_learn default
    for majority_class_undersampling in `seq 0.05 0.05 0.50`; do
      time python loctext/learning/train.py --corpus_percentage 1.0 --minority_class $minority_class --majority_class_undersampling $majority_class_undersampling --svm_hyperparameter_c $c &> $logsdir/"loctext_id${id}_m${minority_class}_u${majority_class_undersampling}_c${c}.log" &
    done
    wait
    for majority_class_undersampling in `seq 0.55 0.05 1`; do
      time python loctext/learning/train.py --corpus_percentage 1.0 --minority_class $minority_class --majority_class_undersampling $majority_class_undersampling --svm_hyperparameter_c $c &> $logsdir/"loctext_id${id}_m${minority_class}_u${majority_class_undersampling}_c${c}.log" &
    done
    wait
  done
done
