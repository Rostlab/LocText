import pickle
from itertools import product
from loctext.util import repo_path, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID
from loctext.util import simple_parse_GO


GO_TREE = simple_parse_GO.simple_parse(repo_path("resources", "ontologies", "go-basic.cellular_component.latest.obo"))
"""
Dictionary with go term child --> to [list of go term parents] relationships
"""


def get_localization_name(go_id, default=""):
    return GO_TREE.get(go_id, (default, "", ""))[0]


def _verify_in_ontology(term):
    if term not in GO_TREE:
        raise KeyError("The term '{}' is not recognized in the considered GO ontology hierarchy".format(term))


def are_go_parent_and_child(parent, child):
    """
    True if terms are equal or parent is indeed a parent in the localization GO of the child. False otherwise.
    """
    return _go_ids_accept_single(parent, child) is True


# ----------------------------------------------------------------------------------------------------

SWISSPROT_ALL_RELATIONS = None
"""
GO Localization/Component annotations written in SwissProt

Dictionary: {Organism ID -> {UniProt AC --> set[GO]} (set of GO ids explicitly written in SwissProt)
"""

with open(repo_path("resources", "features", "SwissProt_all_relations.pickle"), "rb") as f:
    SWISSPROT_ALL_RELATIONS = pickle.load(f)


def is_in_swissprot(uniprot_ac, go, organism_id):
    return is_in_swissprot_explicitly_written(uniprot_ac, go, organism_id) or \
        is_parent_of_swissprot_annotation(uniprot_ac, go, organism_id)


def is_in_swissprot_explicitly_written(uniprot_ac, go, organism_id):
    organism_relations = SWISSPROT_ALL_RELATIONS[organism_id]  # fails with non-supported organisms
    return go in organism_relations.get(uniprot_ac, set())


def is_parent_of_swissprot_annotation(uniprot_ac, go, organism_id):
    organism_relations = SWISSPROT_ALL_RELATIONS[organism_id]  # fails with non-supported organisms
    try:
        return any(are_go_parent_and_child(go, swissprot) for swissprot in organism_relations.get(uniprot_ac, set()))
    except KeyError:
        return False


def is_child_of_swissprot_annotation(uniprot_ac, go, organism_id):
    organism_relations = SWISSPROT_ALL_RELATIONS[organism_id]  # fails with non-supported organisms
    try:
        return any(are_go_parent_and_child(swissprot, go) for swissprot in organism_relations.get(uniprot_ac, set()))
    except KeyError:
        return False


# ----------------------------------------------------------------------------------------------------


def accept_relation_uniprot_go(gold, pred):

    if gold == pred and gold != "":
        return True

    # Note: the | separator is defined by and depends on nalaf

    [_, g_pro_key, g_n_7, g_loc_key, g_n_8] = gold.split("|")
    assert g_pro_key == UNIPROT_NORM_ID, gold
    assert g_loc_key == GO_NORM_ID, gold

    [_, p_pro_key, p_n_7, p_loc_key, p_n_8] = pred.split("|")
    assert p_pro_key == UNIPROT_NORM_ID, pred
    assert p_loc_key == GO_NORM_ID, pred

    uniprot_accept = _accept_uniprot_ids_multiple(g_n_7, p_n_7)
    go_accept = _accept_go_ids_multiple(g_n_8, p_n_8)
    combined = {uniprot_accept, go_accept}

    if combined == {True}:
        return True
    elif False in combined:
        return False
    else:
        return None


def _accept_taxonomy_ids_single(gold, pred):
    return gold == pred


def _accept_uniprot_ids_multiple(gold, pred):
    """
    If all golds are UNKNOWN normalization, return None (reject) else accept if any pair match is equals
    """

    if gold == pred:
        return True

    # see (nalaf) evaluators::_normalized_fun
    golds = [g for g in gold.split(',') if not g.startswith("UNKNOWN:")]
    preds = [p for p in pred.split(',')]

    if not golds:
        return None

    return any(g == p for (g, p) in product(golds, preds))


def _accept_go_ids_multiple(gold, pred):
    """
    Apply essentially same behavior as for multiple unitprot_ids:
    accept if any is true, otherwise None if any is None, or otherwise False
    """
    if gold == pred:
        return True

    # see (nalaf) evaluators::_normalized_fun
    golds = [g for g in gold.split(',') if not g.startswith("UNKNOWN:")]
    preds = [p for p in pred.split(',')]

    if not golds:
        return None

    one_is_None = False

    for (g, p) in product(golds, preds):
        decision = _go_ids_accept_single(g, p)
        if decision is True:
            return True
        elif decision is None:
            one_is_None = True

    if one_is_None:
        return None
    else:
        return False


def _go_ids_accept_single(gold, pred):
    """
    3 outcomes:

    * gold is equal or parent (direct or indirect) of pred --> accept (True)
    * pred is parent (direct or indirect) of gold --> ignore (None)
    * else: no relationship whatsoever --> reject (False)
    """

    if gold == pred:  # Arbitrarily, we do not test whether they belong to the ontology
        return True

    _verify_in_ontology(gold)
    _verify_in_ontology(pred)

    gold_parents = GO_TREE.get(gold).parents
    pred_parents = GO_TREE.get(pred).parents

    if len(gold_parents) == 0:  # gold is root from the start
        return True
    if len(pred_parents) == 0:  # pred is root from the start
        return None

    gold_is_parent_of_pred = _go_ids_accept_single_recursive(gold, pred, pred_parents)
    if gold_is_parent_of_pred:
        return True
    pred_is_parent_of_gold = _go_ids_accept_single_recursive(pred, gold, gold_parents)
    if pred_is_parent_of_gold:
        return None
    else:
        return False


def _go_ids_accept_single_recursive(a, b, b_parents):
    """
    2 outcomes:

    * a is parent (direct or indirect) of b --> True
    * else --> False
    """

    if a == b:
        return True

    return any(_go_ids_accept_single_recursive(a, pp, GO_TREE.get(pp).parents) for pp in b_parents if pp in GO_TREE)


# --------------------------------------------------------------------------------------------------


def accept_entity_uniprot_go_taxonomy(gold, pred):

    if gold == pred and gold != "":
        return True

    [g_class_id, g_offsets, g_norm_id, g_norm_value] = gold.split('|')
    [p_class_id, p_offsets, p_norm_id, p_norm_value] = pred.split('|')

    if g_class_id != p_class_id or g_norm_id != p_norm_id:
        return False

    # Check if the offsets overlap
    if _overlap_entities_offsets(g_offsets, p_offsets):

        if g_norm_value != "" and g_norm_value == p_norm_value:
            return True

        if g_norm_id == UNIPROT_NORM_ID:
            return _accept_uniprot_ids_multiple(g_norm_value, p_norm_value)
        elif g_norm_id == GO_NORM_ID:
            return _accept_go_ids_multiple(g_norm_value, p_norm_value)
        elif g_norm_id == TAXONOMY_NORM_ID:
            return _accept_taxonomy_ids_single(g_norm_value, p_norm_value)
        else:
            raise AssertionError

    else:
        return False


def _overlap_entities_offsets(g_offsets, p_offsets):
    g_start_offset, g_end_offset = g_offsets.split(',')
    p_start_offset, p_end_offset = p_offsets.split(',')

    return int(g_start_offset) < int(p_end_offset) and int(g_end_offset) > int(p_start_offset)


# --------------------------------------------------------------------------------------------------


assert(is_in_swissprot("P51811", "GO:0016020", 9606))
assert(is_in_swissprot("Q53GL0", "GO:0005886", 9606))
