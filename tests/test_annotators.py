# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError: # Parent module '' not loaded, cannot perform relative import
    pass

from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from loctext.learning.train import read_corpus, evaluate_with_argv
from nalaf import print_verbose, print_debug
import math
import sys


def test_baseline():
    corpus = read_corpus("LocText")

    EXPECTED_F = 0.6157
    EXPECTED_F_SE = 0.0028

    annotator_fun = (lambda _: StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID).annotate)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_fun, corpus, evaluator, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure
    print(rel_evaluation)

    return rel_evaluation


def test_LocText(use_full_corpus):
    corpus = read_corpus("LocText")

    if (use_full_corpus):
        EXPECTED_F = 0.5433
        EXPECTED_F_SE = 0.0028
    else:
        corpus, _ = corpus.percentage_split(0.1)
        EXPECTED_F = 0.4906
        EXPECTED_F_SE = 0.0083

    rel_evaluation = evaluate_with_argv(['--corpus_percentage', '0.1'])

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)
    print(rel_evaluation)

    return rel_evaluation


if __name__ == "__main__":
    test_baseline()
    test_LocText(use_full_corpus=('--corpus_percentage 1' in sys.argv))
