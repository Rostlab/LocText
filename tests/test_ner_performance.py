from nalaf.learning.evaluators import MentionLevelEvaluator
from loctext.learning.train import read_corpus

try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass


EVALUATOR = MentionLevelEvaluator(subclass_analysis=True)


def test_get_evaluation_result_of_corpus():
    """
    Evaluates the performance of corpus entities [e_1 (Protein), e_2 (Localization) and e_3 (Organism)]
    [precision, recall and f-measure]
    :param corpus:
    :return:
    """
    # Gets both annotations and pred_annotations entities.
    corpus = read_corpus("LocText", corpus_percentage=1.0, predict_entities=True)

    evaluations = EVALUATOR.evaluate(corpus)
    print("-----------------------------------------------------------------------------------")
    print(evaluations)
    print("-----------------------------------------------------------------------------------")


if __name__ == "__main__":
    test_get_evaluation_result_of_corpus(corpus)
