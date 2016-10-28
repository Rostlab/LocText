# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError: # Parent module '' not loaded, cannot perform relative import
    pass

from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from loctext.learning.annotators import LocTextBaselineRelationExtractor
from loctext.learning.train import read_corpus, train
from nalaf import print_verbose, print_debug
import math
import sys

k_num_folds = 5
use_validation_set = True

def test_baseline():
    corpus = read_corpus("LocText")
    EXPECTED_F = 0.4234297812279464
    EXPECTED_F_SE = 0.0024623653397242064

    annotator_fun = (lambda _: LocTextBaselineRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID))
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_fun, corpus, evaluator, k_num_folds, use_validation_set=use_validation_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)

    return rel_evaluation


def test_LocText(use_full_corpus):
    corpus = read_corpus("LocText")
    print("use_full_corpus: " + str(use_full_corpus))

    if (use_full_corpus):
        EXPECTED_F = 0.5095
        EXPECTED_F_SE = 0.0028
    else:
        corpus, _ = corpus.percentage_split(0.40)
        EXPECTED_F = 0.5984
        EXPECTED_F_SE = 0.0048

    annotator_fun = (lambda train_set: train(train_set, {'use_tk': False}))
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_fun, corpus, evaluator, k_num_folds, use_validation_set=use_validation_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)

    return rel_evaluation


if __name__ == "__main__":
    test_baseline()
    test_LocText(use_full_corpus=('--use-full-corpus' in sys.argv))
