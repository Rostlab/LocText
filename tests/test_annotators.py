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
    # Absolute full: abstracts + full text
    # EXPECTED_F = 0.4234297812279464
    # EXPECTED_F_SE = 0.0024623653397242064

    # Full: abstracts
    EXPECTED_F = 0.4547
    EXPECTED_F_SE = 0.0026

    annotator_fun = (lambda _: LocTextBaselineRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID))
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_fun, corpus, evaluator, k_num_folds, use_validation_set=use_validation_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)

    return rel_evaluation


def test_LocText(use_full_corpus):
    corpus = read_corpus("LocText")

    from nalaf.preprocessing.edges import SimpleEdgeGenerator
    from nalaf.preprocessing.spliters import NLTKSplitter
    from nalaf.preprocessing.tokenizers import TmVarTokenizer

    splitter = NLTKSplitter()
    tokenizer = TmVarTokenizer()
    edger = SimpleEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID)

    splitter.split(corpus)
    tokenizer.tokenize(corpus)
    edger.generate(corpus)
    corpus.label_edges()

    P = 0
    N = 0

    for e in corpus.edges():
        assert e.target != 0, str(e)
        print(e, e.target)
        if e.target > 0:
            P += 1
        else:
            N += 1

    # with all (abstract+fulltext), P=614 vs N=1480
    # with only abstracts -- Corpus size: 100 -- #P=351 vs. #N=308
    print("Corpus size: {} -- #P={} vs. #N={}".format(len(corpus), P, N))

    if (use_full_corpus):
        # 0.1
        # 0.4 0.618421052631579
        # 0.5
        # full 0.5807962529274006
        EXPECTED_F = 0.5433
        EXPECTED_F_SE = 0.0028
    else:
        corpus, _ = corpus.percentage_split(0.1)
        EXPECTED_F = 0.4906
        EXPECTED_F_SE = 0.0083

    annotator_fun = (lambda train_set: train(train_set, {'use_tk': False}))
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_fun, corpus, evaluator, k_num_folds, use_validation_set=use_validation_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)

    return rel_evaluation


if __name__ == "__main__":
    test_baseline()
    test_LocText(use_full_corpus=('--use-full-corpus' in sys.argv))
