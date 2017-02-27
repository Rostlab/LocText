from nalaf.learning.evaluators import EntityLevelEvaluator, MentionLevelEvaluator
from loctext.learning.train import read_corpus
from nalaf.structures.data import Entity
from loctext.learning.evaluations import entity_accept_uniprot_go_taxonomy
from loctext.util import PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID

try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass


def entity_overlap_fun(gold, pred):
    """
    :return:True  - if the offsets of the two entities overlap
            False - otherwise
    """
    return gold.offset < pred.end_offset() and gold.end_offset() > pred.offset


def _get_entity_evaluator(evaluation_level):
    """
    Returns EntityLevelEvaluator object based on specified evaluation_level
    """
    normalization_penalization = "soft"
    ENTITY_OVERLAP_FUN = entity_overlap_fun

    if evaluation_level == 1:
        ENTITY_MAP_FUN = Entity.__repr__
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 2:
        ENTITY_MAP_FUN = 'lowercased'
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 3:
        ENTITY_MAP_FUN = EntityLevelEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization
        )
        ENTITY_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 4:
        ENTITY_MAP_FUN = EntityLevelEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_fun'](
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

    evaluator = EntityLevelEvaluator(
        subclass_analysis=True,
        entity_map_fun=ENTITY_MAP_FUN,
        entity_overlap_fun=ENTITY_OVERLAP_FUN,
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
    # Gets both annotations and pred_annotations entities.
    corpus = read_corpus("LocText", corpus_percentage=1.0, predict_entities=True)

    evaluator = _get_entity_evaluator(evaluation_level=4)
    evaluations = evaluator.evaluate(corpus)
    print("-----------------------------------------------------------------------------------")
    print(evaluations)
    print("-----------------------------------------------------------------------------------")


if __name__ == "__main__":
    test_get_evaluation_result_of_corpus()
