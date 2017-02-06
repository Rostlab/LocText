# Be able to call directly such as `python test_annotators.py`
from build.lib.nalaf.learning.taggers import StubSamePartRelationExtractor

from loctext.learning.annotators import StringTagger

try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass

from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.learning.taggers import StubSameSentenceRelationExtractor, StubRelationExtractor
from loctext.learning.train import read_corpus, evaluate_with_argv
from nalaf import print_verbose, print_debug
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator, CombinatorEdgeGenerator
import math
import sys
from loctext.learning.evaluations import relation_accept_uniprot_go
from nalaf.structures.data import Entity


# See conftest.py too
TEST_MIN_CORPUS_PERCENTAGE = 0.4

EVALUATION_LEVEL = 4

if EVALUATION_LEVEL == 1:
    ENTITY_MAP_FUN = Entity.__repr__
    RELATION_ACCEPT_FUN = str.__eq__
elif EVALUATION_LEVEL == 2:
    ENTITY_MAP_FUN = 'lowercased'
    RELATION_ACCEPT_FUN = str.__eq__
elif EVALUATION_LEVEL == 3:
    ENTITY_MAP_FUN = 'normalized_first'
    RELATION_ACCEPT_FUN = str.__eq__
elif EVALUATION_LEVEL == 4:
    ENTITY_MAP_FUN = 'normalized_first'
    RELATION_ACCEPT_FUN = relation_accept_uniprot_go

EVALUATOR = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, entity_map_fun=ENTITY_MAP_FUN, relation_accept_fun=RELATION_ACCEPT_FUN)


def test_baseline_SS(corpus_percentage):
    corpus = read_corpus("LocText", corpus_percentage)

    if (corpus_percentage == 1.0):
        # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
        # r_5	241	106	90	0	0	0.6945	0.7281	0.7109	0.0028	0.6945	0.7281	0.7109	0.0028
        # Computation(precision=0.6945244956772334, precision_SE=0.0028956219539813754, recall=0.7280966767371602, recall_SE=0.004139235568395008, f_measure=0.7109144542772862, f_measure_SE=0.002781031509621811)
        EXPECTED_F = 0.7109
        EXPECTED_F_SE = 0.0028
    else:
        # Computation(precision=0.7657657657657657, precision_SE=0.004062515118259012, recall=0.6640625, recall_SE=0.006891900506329359, f_measure=0.7112970711297071, f_measure_SE=0.004544881638992179)
        EXPECTED_F = 0.7113
        EXPECTED_F_SE = 0.0046

    annotator_gen_fun = (lambda _: StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID).annotate)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, EVALUATOR, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure
    print("SS Baseline", rel_evaluation)

    return rel_evaluation



def test_LocText_SS(corpus_percentage):

    if (corpus_percentage == 1.0):
        # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
        # r_5	234	132	212	0	0	0.6393	0.5247	0.5764	0.0033	0.6393	0.5247	0.5764	0.0031
        EXPECTED_F = 0.6178
        EXPECTED_F_SE = 0.0027
    else:
        EXPECTED_F = 0.6779
        EXPECTED_F_SE = 0.0045

    _test_LocText(corpus_percentage, model='SS', EXPECTED_F=EXPECTED_F)


def test_baseline_DS(corpus_percentage):
    corpus = read_corpus("LocText", corpus_percentage)

    if corpus_percentage == 1.0:
        EXPECTED_F = 0.6137
        EXPECTED_F_SE = 0.0027
    else:
        EXPECTED_F = 0.6483
        EXPECTED_F_SE = 0.0034

    edge_generator = SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=1)
    annotator_gen_fun = (lambda _: StubRelationExtractor(edge_generator).annotate)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, EVALUATOR, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure
    print("DS Baseline", rel_evaluation)

    return rel_evaluation


def test_LocText_DS(corpus_percentage):

    if (corpus_percentage == 1.0):
        EXPECTED_F = 0.4094
        EXPECTED_F_SE = 0.0024
    else:
        EXPECTED_F = 0.4301
        EXPECTED_F_SE = 0.0043

    _test_LocText(corpus_percentage, model='DS', EXPECTED_F=EXPECTED_F)


def test_baseline_Combined(corpus_percentage):
    corpus = read_corpus("LocText", corpus_percentage)

    if corpus_percentage == 1.0:
        EXPECTED_F = 0.6652
        EXPECTED_F_SE = 0.0026
    else:
        EXPECTED_F = 0.6918
        EXPECTED_F_SE = 0.0031

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

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure
    print("DS Baseline", rel_evaluation)

    return rel_evaluation


# Note: would be way better to be able to reuse the already trained models in the other tests methods
def test_LocText_Combined(corpus_percentage):

    if (corpus_percentage == 1.0):
        EXPECTED_F = 0.5734
        EXPECTED_F_SE = 0.0024
    else:
        EXPECTED_F = 0.6667
        EXPECTED_F_SE = 0.0027

    _test_LocText(corpus_percentage, model='Combined', EXPECTED_F=EXPECTED_F)


def _test_LocText(corpus_percentage, model, EXPECTED_F=None, EXPECTED_F_SE=0.001):
    # Note: EXPECTED_F=None will make the test fail for non-yet verified evaluations
    # Note: the real StdErr's are around ~0.0027-0.0095. Decrease them by default to be more strict with tests

    assert corpus_percentage in [TEST_MIN_CORPUS_PERCENTAGE, 1.0], "corpus_percentage must == {} or 1.0. You gave: {}".format(str(TEST_MIN_CORPUS_PERCENTAGE), str(corpus_percentage))

    corpus = read_corpus("LocText", corpus_percentage)
    # add '--evaluation_level', '1' since this argument is required
    rel_evaluation = evaluate_with_argv(['--corpus_percentage', str(corpus_percentage), '--model', model])

    print("LocText " + model, rel_evaluation)
    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)

    return rel_evaluation


# Test case for baseline with prediction_annotations from StringTagger.
def test_baseline_full(corpus_percentage):
    corpus = read_corpus("LocText", 0.02)

    if corpus_percentage == 1.0:
        EXPECTED_F = 0.6652
        EXPECTED_F_SE = 0.0026
    else:
        EXPECTED_F = 0.6918
        EXPECTED_F_SE = 0.0031

    StringTagger(False, PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID).annotate(corpus)

    annotator_gen_fun = (lambda _: StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID).annotate)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, EVALUATOR, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure
    print("Full Baseline", rel_evaluation)


# def test_loctext_full():


if __name__ == "__main__":

    test_baseline()

    corpus_percentage = float(sys.argv[2]) if len(sys.argv) == 3 else TEST_MIN_CORPUS_PERCENTAGE
    test_LocText_SS(corpus_percentage)
