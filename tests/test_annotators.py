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
    BASELINE_F_ON_LOCTEXT = 0.5566600397614313
    BASELINE_F_SE_ON_LOCTEXT = 0.006041640890651582

    train_set, test_set = corpus.percentage_split()

    pipeline, svmlight = train(train_set, {'use_tk': False})

    annotator = LocTextRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, svmlight.model_path, svmlight, pipeline)
    annotator.annotate(test_set)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    results = evaluator.evaluate(test_set)

    rel_evaluation = results(REL_PRO_LOC_ID).compute(strictness="exact")

    assert math.isclose(rel_evaluation.f_measure, BASELINE_F_ON_LOCTEXT)
    assert math.isclose(rel_evaluation.f_measure_SE, BASELINE_F_SE_ON_LOCTEXT, rel_tol=0.1)

    return rel_evaluation


if __name__ == "__main__":
    test_baseline()
    test_LocText()
