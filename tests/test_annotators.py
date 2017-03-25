from loctext.learning.annotators import LocTextAnnotator

try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    raise

from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.learning.taggers import StubSameSentenceRelationExtractor, StubRelationExtractor, StubSamePartRelationExtractor
from loctext.learning.train import read_corpus, evaluate_with_argv, get_evaluator
from nalaf import print_verbose, print_debug
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator, CombinatorEdgeGenerator
import math
import sys
from loctext.learning.evaluations import accept_relation_uniprot_go
from nalaf.structures.data import Entity


# See conftest.py too
TEST_MIN_CORPUS_PERCENTAGE = 0.4

EVALUATION_LEVEL = 4

EVALUATOR = get_evaluator(EVALUATION_LEVEL, evaluate_only_on_edges_plausible_relations=False, normalization_penalization="no")


# -----------------------------------------------------------------------------------


def test_baseline_D0(evaluation_level, corpus_percentage):
    if (corpus_percentage == 1.0):
        EXPECTED_F = 0.7248
    else:
        EXPECTED_F = 0.7113

    corpus = read_corpus("LocText", corpus_percentage)

    annotator_gen_fun = (lambda _: StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID).annotate)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, EVALUATOR, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    print(rel_evaluation)
    print(evaluations)
    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=0.001 * 1.1), rel_evaluation.f_measure

    return evaluations


def test_LocText_D0(corpus_percentage):

    if (corpus_percentage == 1.0):
        EXPECTED_F = 0.7945
    else:
        EXPECTED_F = None

    _test_LocText(corpus_percentage, model='D0', EXPECTED_F=EXPECTED_F)


# -----------------------------------------------------------------------------------


def test_baseline_D1(corpus_percentage):
    corpus = read_corpus("LocText", corpus_percentage)

    if corpus_percentage == 1.0:
        EXPECTED_F = 0.6137
    else:
        EXPECTED_F = 0.6483

    edge_generator = SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=1)
    annotator_gen_fun = (lambda _: StubRelationExtractor(edge_generator).annotate)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, EVALUATOR, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(evaluations.f_measure, EXPECTED_F, abs_tol=0.001 * 1.1), rel_evaluation.f_measure
    print(rel_evaluation)

    return evaluations


def test_LocText_D1(corpus_percentage):

    if (corpus_percentage == 1.0):
        EXPECTED_F = 0.4094
    else:
        EXPECTED_F = 0.4301

    _test_LocText(corpus_percentage, model='D1', EXPECTED_F=EXPECTED_F)


# -----------------------------------------------------------------------------------


def test_baseline_D0_D1(corpus_percentage):
    corpus = read_corpus("LocText", corpus_percentage)

    if corpus_percentage == 1.0:
        EXPECTED_F = 0.6652
    else:
        EXPECTED_F = 0.6918

    edge_generator = CombinatorEdgeGenerator(
        SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=0, rewrite_edges=False),
        SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=1, rewrite_edges=False),  # Recall: 88.52
        # SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=2, rewrite_edges=False),
        # SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=3, rewrite_edges=False),  #
        # SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=4, rewrite_edges=False),
        # SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=5, rewrite_edges=False),  #
        # SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=6, rewrite_edges=False),  # Recall: 99.70
    )

    annotator_gen_fun = (lambda _: StubRelationExtractor(edge_generator).annotate)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, EVALUATOR, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=0.001 * 1.1), rel_evaluation.f_measure
    print("D1 Baseline", rel_evaluation)

    return rel_evaluation


# Note: would be way better to be able to reuse the already trained models in the other tests methods
def test_LocText_D0_D1(corpus_percentage):

    if (corpus_percentage == 1.0):
        EXPECTED_F = 0.5734
    else:
        EXPECTED_F = 0.6667

    _test_LocText(corpus_percentage, model='D0,D1', EXPECTED_F=EXPECTED_F)


# -----------------------------------------------------------------------------------


# "Full" as in the full pipeline: first ner, then re
def test_baseline_full(corpus_percentage):
    if (corpus_percentage == 1.0):
        EXPECTED_F = 0.4712
    else:
        EXPECTED_F = None

    corpus = read_corpus("LocText", corpus_percentage, predict_entities="9606,3702,4932")

    annotator_gen_fun = (lambda _: StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, use_gold=False, use_pred=True).annotate)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, EVALUATOR, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    print(evaluations)
    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=0.001 * 1.1), rel_evaluation.f_measure

    return evaluations


# "Full" as in the full pipeline: first ner, then re
def test_loctext_full(corpus_percentage):
    if (corpus_percentage == 1.0):
        EXPECTED_F = 0.6178
    else:
        EXPECTED_F = 0.6779

    _test_LocText(corpus_percentage, model='D0', EXPECTED_F=EXPECTED_F, predict_entities=True)


# -----------------------------------------------------------------------------------


def _test_LocText(corpus_percentage, model, EXPECTED_F=None, predict_entities=False, EXPECTED_F_SE=0.001):
    # Note: EXPECTED_F=None will make the test fail for non-yet verified evaluations
    # Note: the real StdErr's are around ~0.0027-0.0095. Decrease them by default to be more strict with tests

    assert corpus_percentage in [TEST_MIN_CORPUS_PERCENTAGE, 1.0], "corpus_percentage must == {} or 1.0. You gave: {}".format(str(TEST_MIN_CORPUS_PERCENTAGE), str(corpus_percentage))

    corpus = read_corpus("LocText", corpus_percentage, predict_entities)

    rel_evaluation = evaluate_with_argv(['--model', model, '--corpus_percentage', str(corpus_percentage), '--evaluation_level', str(EVALUATION_LEVEL), '--predict_entities', 'true'])

    print("LocText " + model, rel_evaluation)
    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)

    return rel_evaluation


# -----------------------------------------------------------------------------------


if __name__ == "__main__":

    test_baseline()

    corpus_percentage = float(sys.argv[2]) if len(sys.argv) == 3 else TEST_MIN_CORPUS_PERCENTAGE
    test_LocText_D0(corpus_percentage)
