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


# See conftest.py too
TEST_MIN_CORPUS_PERCENTAGE = 0.4


def test_baseline():
    corpus = read_corpus("LocText")

    # Baseline Computation(precision=0.6077586206896551, precision_SE=0.002781236497769688, recall=0.6238938053097345, recall_SE=0.004052746798458239, f_measure=0.6157205240174672, f_measure_SE=0.002826453358117873)
    EXPECTED_F = 0.6157
    EXPECTED_F_SE = 0.0028

    annotator_gen_fun = (lambda _: StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID).annotate)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, evaluator, k_num_folds=5, use_validation_set=True)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, EXPECTED_F, abs_tol=EXPECTED_F_SE * 1.1), rel_evaluation.f_measure
    print("Baseline", rel_evaluation)

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


# def test_LocText_SS(corpus_percentage):
#
#     if (corpus_percentage == 1.0):
#         # Computation(precision=0.6624365482233503, precision_SE=0.0029261497595035445, recall=0.5787139689578714, recall_SE=0.004036629092741261, f_measure=0.6177514792899409, f_measure_SE=0.0027412422752843557)
#         EXPECTED_F = 0.6178
#         EXPECTED_F_SE = 0.0027
#     else:
#         # Computation(precision=0.7426470588235294, precision_SE=0.004447039958950779, recall=0.6121212121212121, recall_SE=0.005972946581336089, f_measure=0.6710963455149502, f_measure_SE=0.0043836182031360155)
#         EXPECTED_F = 0.6711
#         EXPECTED_F_SE = 0.0043
#
#     _test_LocText(corpus_percentage, model='SS', EXPECTED_F=EXPECTED_F)


def test_LocText_DS(corpus_percentage):

    # So far DS model does nothing (at least, it _should_ to nothing)
    EXPECTED_F = 0.1

    _test_LocText(corpus_percentage, model='DS', EXPECTED_F=EXPECTED_F)


# # TODO find the way to do the training only once for each submodel
# def test_LocText_Combined(corpus_percentage):
#
#     # Should be now the same as SS results, since the DS model, as of now, _SHOULD_ DO NOTHING
#     if (corpus_percentage == 1.0):
#         EXPECTED_F = 0.6178
#         EXPECTED_F_SE = 0.0027
#     else:
#         EXPECTED_F = 0.6711
#         EXPECTED_F_SE = 0.0043
#
#     _test_LocText(corpus_percentage, model='Combined', EXPECTED_F=EXPECTED_F)


if __name__ == "__main__":

    test_baseline()

    corpus_percentage = float(sys.argv[2]) if len(sys.argv) == 3 else TEST_MIN_CORPUS_PERCENTAGE
    test_LocText_SS(corpus_percentage)
