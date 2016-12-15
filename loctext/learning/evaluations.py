from loctext.util import repo_path
from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path, UNIPROT_NORM_ID, GO_NORM_ID
from loctext.util import simple_parse_GO
from itertools import product


GO_TREE = simple_parse_GO.simple_parse(repo_path(["resources", "ontologies", "go-basic.cellular_component.latest.obo"]))
"""
Dictionary with go term child --> to [list of go term parents] relationships
"""

def relation_accept_uniprot_go(gold, pred):

    if gold == pred and gold != "":
        return True

    # Note: the | separator is defined by and depends on nalaf

    [_, g_pro_key, g_n_7, g_loc_key, g_n_8] = gold.split("|")
    assert g_pro_key == UNIPROT_NORM_ID
    assert g_loc_key == GO_NORM_ID

    [_, p_pro_key, p_n_7, p_loc_key, p_n_8] = pred.split("|")
    assert p_pro_key == UNIPROT_NORM_ID
    assert p_loc_key == GO_NORM_ID

    return _uniprot_ids_accept(g_n_7, p_n_7) and _go_ids_accept(g_n_8, p_n_8)


def _uniprot_ids_accept(gold, pred):

    if gold == pred:
        return True

    return any(g == p for (g, p) in product(gold.split(','), pred.split(',')))


def _verify_in_ontology(term):

    if term not in GO_TREE:
        raise KeyError("The term '{}' is not recognized in the considered GO ontology hierarchy".format(term))


def _go_ids_accept(gold, pred):
    """
    3 outcomes:

    * gold is parent (direct or indirect) of pred --> accept (True)
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

    gold_is_parent_of_pred = _go_ids_accept_recursive(gold, pred, pred_parents)
    if gold_is_parent_of_pred:
        return True
    pred_is_parent_of_gold = _go_ids_accept_recursive(pred, gold, gold_parents)
    if pred_is_parent_of_gold:
        return None
    else:
        return False


def _go_ids_accept_recursive(a, b, b_parents):
    """
    2 outcomes:

    * a is parent (direct or indirect) of b --> True
    * else --> False
    """

    if a == b:
        return True

    return any(_go_ids_accept_recursive(a, pp, GO_TREE.get(pp).parents) for pp in b_parents if pp in GO_TREE)
