# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError: # Parent module '' not loaded, cannot perform relative import
    pass

from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from loctext.learning.annotators import LocTextBaselineRelationExtractor, LocTextRelationExtractor
from loctext.learning.train import read_corpus, train
import math


k_num_folds = 5
use_validation_set = True

def test_baseline():
    corpus = read_corpus("LocText")
    BASELINE_F_ON_LOCTEXT = 0.4234297812279464
    BASELINE_F_SE_ON_LOCTEXT = 0.0024623653397242064

    annotator = LocTextBaselineRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator, corpus, evaluator, k_num_folds, use_validation_set=use_validation_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, BASELINE_F_ON_LOCTEXT)
    assert math.isclose(rel_evaluation.f_measure_SE, BASELINE_F_SE_ON_LOCTEXT, rel_tol=0.1)

    return rel_evaluation


def test_LocText():
    corpus = read_corpus("LocText")
    BASELINE_F_ON_LOCTEXT = 0.5819397993311036
    BASELINE_F_SE_ON_LOCTEXT = 0.005016402372379795

    train_set, test_set = corpus.percentage_split()

    pipeline, svmlight = train(train_set, {'use_tk': False})

    annotator = LocTextRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, svmlight.model_path, pipeline=pipeline, svmlight=svmlight)
    annotator.annotate(test_set)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    results = evaluator.evaluate(test_set)

    rel_evaluation = results(REL_PRO_LOC_ID).compute(strictness="exact")

    # # class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
    # r_5	174	167	83	0	0	0.5103	0.6770	0.5819	0.0049	0.5103	0.6770	0.5819	0.0051
    print(results)
    #Computation(precision=0.5102639296187683, precision_SE=0.004894534246666679, recall=0.6770428015564203, recall_SE=0.007291120408974056, f_measure=0.5819397993311036, f_measure_SE=0.005016402372379795)
    print(rel_evaluation)

    assert math.isclose(rel_evaluation.f_measure, BASELINE_F_ON_LOCTEXT)
    assert math.isclose(rel_evaluation.f_measure_SE, BASELINE_F_SE_ON_LOCTEXT, rel_tol=0.1)

    return rel_evaluation


if __name__ == "__main__":
    test_baseline()
    test_LocText()
