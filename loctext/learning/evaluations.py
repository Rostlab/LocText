from loctext.util import repo_path
from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path, UNIPROT_NORM_ID, GO_NORM_ID
from loctext.util import simple_parse_GO


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
    # Temporal
    return gold == pred


def _go_ids_equiv(gold, pred):
    """
    the gold go term must be the parent to accept the go prediction, not the other way around
    """

    if gold == pred:
        return True

    gold_is_parent_of_pred = gold in GO_TREE[pred]

    print(gold, pred, GO_TREE[pred], gold_is_parent_of_pred)

    return gold_is_parent_of_pred
