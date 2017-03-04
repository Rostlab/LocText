from nalaf.learning.evaluators import EntityEvaluator, MentionLevelEvaluator
from loctext.learning.train import read_corpus
from nalaf.structures.data import Entity
from loctext.learning.evaluations import entity_accept_uniprot_go_taxonomy
from loctext.util import PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID
import pytest

try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    raise
    # pass


def test_get_evaluation_result_of_corpus(evaluation_level):
    """
    Evaluates the performance of corpus entities [e_1 (Protein), e_2 (Localization) and e_3 (Organism)]
    [precision, recall and f-measure]
    :param corpus:
    :return:
    """

    # Gets both annotation and pred_annotation entities.
    corpus = read_corpus("LocText", corpus_percentage=1.0, predict_entities=True)

    (mention_evaluator, entity_evaluator) = _get_entity_evaluator(evaluation_level)

    print()
    print("EVALUATION LEVEL:", evaluation_level)
    print()

    print("-----------------------------------------------------------------------------------")
    print("MentionLevelEvaluator")
    print(mention_evaluator.evaluate(corpus))
    print("-----------------------------------------------------------------------------------")
    print()
    print()
    print("-----------------------------------------------------------------------------------")
    print("EntityEvaluator")
    print(entity_evaluator.evaluate(corpus))
    print("-----------------------------------------------------------------------------------")


def _get_entity_evaluator(evaluation_level):
    """
    Returns EntityEvaluator object based on specified evaluation_level
    """
    normalization_penalization = "soft"
    norm_required = True
    norm_not_required = False

    if evaluation_level == 1:
        ENTITY_MAP_FUN = Entity.__repr__
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 2:
        ENTITY_MAP_FUN = 'lowercased'
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 3:
        ENTITY_MAP_FUN = EntityEvaluator.COMMON_ENTITY_MAP_FUNS['entity_normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization,
            normalization_required=norm_not_required
        )
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 4:
        ENTITY_MAP_FUN = EntityEvaluator.COMMON_ENTITY_MAP_FUNS['entity_normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization,
            normalization_required=norm_required
        )
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 5:
        ENTITY_MAP_FUN = EntityEvaluator.COMMON_ENTITY_MAP_FUNS['entity_normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization,
            normalization_required=norm_not_required
        )
        ENTITY_ACCEPT_FUN = entity_accept_uniprot_go_taxonomy

    elif evaluation_level == 6:
        ENTITY_MAP_FUN = EntityEvaluator.COMMON_ENTITY_MAP_FUNS['entity_normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization,
            normalization_required=norm_required
        )
        ENTITY_ACCEPT_FUN = entity_accept_uniprot_go_taxonomy

    else:
        raise AssertionError(evaluation_level)

    entity_evaluator = EntityEvaluator(
        subclass_analysis=True,
        entity_map_fun=ENTITY_MAP_FUN,
        entity_accept_fun=ENTITY_ACCEPT_FUN
    )

    mention_evaluator = MentionLevelEvaluator(subclass_analysis=True)

    return (mention_evaluator, entity_evaluator)


if __name__ == "__main__":
    import sys

    evaluation_level = int(sys.argv[1])

    test_get_evaluation_result_of_corpus(evaluation_level)
