from .context import loctext

from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from loctext.learning.annotators import LocTextBaselineRelationExtractor
from loctext.learning.train import read_corpus
import math


def test_baseline():
    corpus = read_corpus("LocText")
    k_num_folds = 5
    use_validation_set = True
    BASELINE_F_ON_LOCTEXT = 0.4234297812279464
    BASELINE_F_SE_ON_LOCTEXT = 0.0024623653397242064

    annotator = LocTextBaselineRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator, corpus, evaluator, k_num_folds, use_validation_set=use_validation_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, BASELINE_F_ON_LOCTEXT)
    assert math.isclose(rel_evaluation.f_measure_SE, BASELINE_F_SE_ON_LOCTEXT, rel_tol=0.1)
