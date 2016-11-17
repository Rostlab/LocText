from loctext.util import repo_path
from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path, UNIPROT_NORM_ID, GO_NORM_ID
from loctext.util import simple_parse_GO
from itertools import product


GO_TREE = simple_parse_GO.simple_parse(repo_path(["resources", "ontologies", "go-basic.cellular_component.latest.obo"]))
"""
Dictionary with go term child --> to [list of go term parents] relationships
"""

def relation_equals_uniprot_go(gold, pred):

    if gold == pred and gold != "":
        return True

    # Note: the | separator is defined by and depends on nalaf

    [_, g_pro_key, g_n_7, g_loc_key, g_n_8] = gold.split("|")
    assert g_pro_key == UNIPROT_NORM_ID
    assert g_loc_key == GO_NORM_ID

    [_, p_pro_key, p_n_7, p_loc_key, p_n_8] = pred.split("|")
    assert p_pro_key == UNIPROT_NORM_ID
    assert p_loc_key == GO_NORM_ID

    return _uniprot_ids_equiv(g_n_7, p_n_7) and _go_ids_equiv(g_n_8, p_n_8)


def _uniprot_ids_equiv(gold, pred):

    if gold == pred:
        return True

    return any(g == p for (g, p) in product(gold.split(','), pred.split(',')))


def _verify_in_ontology(term):

    if term not in GO_TREE:
        raise KeyError("The term '{}' is not recognized in the considered GO ontology hierarchy".format(term))


def _go_ids_equiv(gold, pred):
    """
    the gold go term must be the parent to accept the go prediction, not the other way around
    """

    if gold == pred:
        return True

    _verify_in_ontology(gold)
    _verify_in_ontology(pred)

    gold_is_root = len(GO_TREE.get(gold)) == 0

    if gold_is_root:
        return True

    pred_parents = GO_TREE.get(pred)

    # direct parent or indirect (recursive) parent
    return gold in pred_parents or any(_go_ids_equiv(gold, pp) for pp in pred_parents if pp in GO_TREE)
