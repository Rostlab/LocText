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
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf.preprocessing.tokenizers import NLTK_TOKENIZER


# See conftest.py too
TEST_MIN_CORPUS_PERCENTAGE = 0.4


def test_SS_baseline():
    corpus = read_corpus("LocText")

    # Computation(precision=0.6083150984682714, precision_SE=0.002974704942625582, recall=0.6233183856502242, recall_SE=0.004130201948613626, f_measure=0.6157253599114065, f_measure_SE=0.0030062001054202924)
    EXPECTED_F = 0.6157  # 62
    EXPECTED_F_SE = 0.0030

    annotator_gen_fun = (lambda _: StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID).annotate)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID)

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
        # Computation(precision=0.753731343283582, precision_SE=0.004064280002070562, recall=0.6158536585365854, recall_SE=0.006147889829984894, f_measure=0.6778523489932885, f_measure_SE=0.004461978434634661)
        EXPECTED_F = 0.6779
        EXPECTED_F_SE = 0.0045

    _test_LocText(corpus_percentage, model='SS', EXPECTED_F=EXPECTED_F)


def test_DS_baseline():
    corpus = read_corpus("LocText")

    # Computation(precision=0.4196969696969697, precision_SE=0.0024518002206926812, recall=0.6210762331838565, recall_SE=0.0038995731275083797, f_measure=0.5009041591320074, f_measure_SE=0.002367560805132471)
    EXPECTED_F = 0.5009
    EXPECTED_F_SE = 0.0024

    edge_generator = SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=1)

    annotator_gen_fun = (lambda _: StubRelationExtractor(edge_generator).annotate)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID)

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
        # Computation(precision=0.5217391304347826, precision_SE=0.004661534209213446, recall=0.36585365853658536, recall_SE=0.00518652082659298, f_measure=0.43010752688172044, f_measure_SE=0.004278857080197886)
        EXPECTED_F = 0.4301
        EXPECTED_F_SE = 0.0043

    _test_LocText(corpus_percentage, model='DS', EXPECTED_F=EXPECTED_F)


# Note: would be way better to be able to reuse the already trained models in the other tests methods
def test_LocText_Combined(corpus_percentage):

    if (corpus_percentage == 1.0):
        # Computation(precision=0.4774381368267831, precision_SE=0.002812150029056132, recall=0.7177242888402626, recall_SE=0.0038763518483362733, f_measure=0.5734265734265734, f_measure_SE=0.002422393130974523)
        EXPECTED_F = 0.5734
        EXPECTED_F_SE = 0.0024
    else:
        # Computation(precision=0.5887850467289719, precision_SE=0.0033947585471817603, recall=0.7682926829268293, recall_SE=0.004128234454709121, f_measure=0.6666666666666666, f_measure_SE=0.0027263757211551205)
        EXPECTED_F = 0.6667
        EXPECTED_F_SE = 0.0027

    _test_LocText(corpus_percentage, model='Combined', EXPECTED_F=EXPECTED_F)


if __name__ == "__main__":

    test_baseline()

    corpus_percentage = float(sys.argv[2]) if len(sys.argv) == 3 else TEST_MIN_CORPUS_PERCENTAGE
    test_LocText_SS(corpus_percentage)
