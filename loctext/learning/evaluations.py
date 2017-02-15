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
    assert g_pro_key == UNIPROT_NORM_ID, gold
    assert g_loc_key == GO_NORM_ID, gold

    [_, p_pro_key, p_n_7, p_loc_key, p_n_8] = pred.split("|")
    assert p_pro_key == UNIPROT_NORM_ID, pred
    assert p_loc_key == GO_NORM_ID, pred

    uniprot_accept = _uniprot_ids_accept_multiple(g_n_7, p_n_7)
    go_accept = _go_ids_accept_multiple(g_n_8, p_n_8)
    combined = {uniprot_accept, go_accept}

    if combined == {True}:
        return True
    elif False in combined:
        return False
    else:
        return None


def _uniprot_ids_accept_multiple(gold, pred):
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


def _go_ids_accept_multiple(gold, pred):
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


def _verify_in_ontology(term):

    if term not in GO_TREE:
        raise KeyError("The term '{}' is not recognized in the considered GO ontology hierarchy".format(term))


def are_go_parent_and_child(parent, child):
    """
    True if parent is indeed a parent in the localization GO of the child. False otherwise.
    """
    return _go_ids_accept_single(parent, child) is True


def _go_ids_accept_single(gold, pred):
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
