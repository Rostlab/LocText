# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError: # Parent module '' not loaded, cannot perform relative import
    pass

from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.learning.taggers import StubSameSentenceRelationExtractor, StubRelationExtractor
from loctext.learning.train import read_corpus, evaluate_with_argv
from nalaf import print_verbose, print_debug
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator
import math
import sys


# See conftest.py too
TEST_MIN_CORPUS_PERCENTAGE = 0.4


def test_SS_baseline():
    corpus = read_corpus("LocText")

    # Baseline Computation(precision=0.6077586206896551, precision_SE=0.002781236497769688, recall=0.6238938053097345, recall_SE=0.004052746798458239, f_measure=0.6157205240174672, f_measure_SE=0.002826453358117873)
    EXPECTED_F = 0.6157
    EXPECTED_F_SE = 0.0028

    annotator_gen_fun = (lambda _: StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID).annotate)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, evaluator, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure
    print("SS Baseline", rel_evaluation)

    return rel_evaluation


def _test_LocText(corpus_percentage, model, EXPECTED_F=None, EXPECTED_F_SE=0.001):
    # Note: EXPECTED_F=None will make the test fail for non-yet verified evaluations
    # Note: the real StdErr's are around ~0.0027-0.0095. Decrease them by default to be more strict with tests

    assert corpus_percentage in [TEST_MIN_CORPUS_PERCENTAGE, 1.0], "corpus_percentage must == {} or 1.0. You gave: {}".format(str(TEST_MIN_CORPUS_PERCENTAGE), str(corpus_percentage))

    corpus = read_corpus("LocText", corpus_percentage)

    rel_evaluation = evaluate_with_argv(['--corpus_percentage', str(corpus_percentage), '--model', model])

    print("LocText " + model, rel_evaluation)
    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1)

    return rel_evaluation


def test_LocText_SS(corpus_percentage):

    if (corpus_percentage == 1.0):
        # Computation(precision=0.6624365482233503, precision_SE=0.0029261497595035445, recall=0.5787139689578714, recall_SE=0.004036629092741261, f_measure=0.6177514792899409, f_measure_SE=0.0027412422752843557)
        EXPECTED_F = 0.6178
        EXPECTED_F_SE = 0.0027
    else:
        # Computation(precision=0.7426470588235294, precision_SE=0.004447039958950779, recall=0.6121212121212121, recall_SE=0.005972946581336089, f_measure=0.6710963455149502, f_measure_SE=0.0043836182031360155)
        EXPECTED_F = 0.6711
        EXPECTED_F_SE = 0.0043

    _test_LocText(corpus_percentage, model='SS', EXPECTED_F=EXPECTED_F)


def test_DS_baseline():
    corpus = read_corpus("LocText")

    # Computation(precision=0.4149560117302053, precision_SE=0.002457395067634639, recall=0.6206140350877193, recall_SE=0.0037411116863785665, f_measure=0.49736379613356774, f_measure_SE=0.00233567117305811)
    EXPECTED_F = 0.4974
    EXPECTED_F_SE = 0.0023

    edge_generator = SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=1)

    annotator_gen_fun = (lambda _: StubRelationExtractor(edge_generator).annotate)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, evaluator, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure
    print("DS Baseline", rel_evaluation)

    return rel_evaluation


def test_LocText_DS(corpus_percentage):

    if (corpus_percentage == 1.0):
        # Computation(precision=0.41743119266055045, precision_SE=0.0030117270329129047, recall=0.40176600441501104, recall_SE=0.0032167797183620065, f_measure=0.4094488188976378, f_measure_SE=0.0025398561212811644)
        EXPECTED_F = 0.4094
        EXPECTED_F_SE = 0.0024
    else:
        # Computation(precision=0.5254237288135594, precision_SE=0.004919604548886615, recall=0.3712574850299401, recall_SE=0.00506665114212087, f_measure=0.43508771929824563, f_measure_SE=0.004266620575873462)
        EXPECTED_F = 0.4351
        EXPECTED_F_SE = 0.0043

    _test_LocText(corpus_percentage, model='DS', EXPECTED_F=EXPECTED_F)


# Note: would be way better to be able to reuse the already trained models in the other tests methods
def test_LocText_Combined(corpus_percentage):

    if (corpus_percentage == 1.0):
        # Computation(precision=0.4774381368267831, precision_SE=0.002812150029056132, recall=0.7177242888402626, recall_SE=0.0038763518483362733, f_measure=0.5734265734265734, f_measure_SE=0.002422393130974523)
        EXPECTED_F = 0.5734
        EXPECTED_F_SE = 0.0024
    else:
        # Computation(precision=0.5844748858447488, precision_SE=0.0032793659586925978, recall=0.7710843373493976, recall_SE=0.004115510298431316, f_measure=0.6649350649350649, f_measure_SE=0.002684996682841366)
        EXPECTED_F = 0.6649
        EXPECTED_F_SE = 0.0027

    _test_LocText(corpus_percentage, model='Combined', EXPECTED_F=EXPECTED_F)


if __name__ == "__main__":

    test_baseline()

    corpus_percentage = float(sys.argv[2]) if len(sys.argv) == 3 else TEST_MIN_CORPUS_PERCENTAGE
    test_LocText_SS(corpus_percentage)
