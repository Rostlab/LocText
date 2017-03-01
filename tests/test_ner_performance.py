from nalaf.learning.evaluators import EntityEvaluator
from loctext.learning.train import read_corpus
from nalaf.structures.data import Entity
from loctext.learning.evaluations import entity_accept_uniprot_go_taxonomy
from loctext.util import PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID

try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass


def _get_entity_evaluator(evaluation_level):
    """
    Returns EntityEvaluator object based on specified evaluation_level
    """
    normalization_penalization = "soft"

    if evaluation_level == 1:
        ENTITY_MAP_FUN = 'lowercased'
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 2:
        ENTITY_MAP_FUN = EntityEvaluator.COMMON_ENTITY_MAP_FUNS['entity_normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization
        )
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 3:
        ENTITY_MAP_FUN = EntityEvaluator.COMMON_ENTITY_MAP_FUNS['entity_normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization
        )
        ENTITY_ACCEPT_FUN = entity_accept_uniprot_go_taxonomy

    else:
        raise AssertionError(evaluation_level)

    evaluator = EntityEvaluator(
        subclass_analysis=True,
        entity_map_fun=ENTITY_MAP_FUN,
        entity_accept_fun=ENTITY_ACCEPT_FUN
    )

    return evaluator


def test_get_evaluation_result_of_corpus():
    """
    Evaluates the performance of corpus entities [e_1 (Protein), e_2 (Localization) and e_3 (Organism)]
    [precision, recall and f-measure]
    :param corpus:
    :return:
    """

    # Gets both annotation and pred_annotation entities.
    corpus = read_corpus("LocText", corpus_percentage=1.0, predict_entities=True)

    evaluator = _get_entity_evaluator(evaluation_level=3)
    evaluations = evaluator.evaluate(corpus)
    print("-----------------------------------------------------------------------------------")
    print(evaluations)
    print("-----------------------------------------------------------------------------------")


if __name__ == "__main__":
    test_get_evaluation_result_of_corpus()
