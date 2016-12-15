# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass

from pytest import raises
from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path, UNIPROT_NORM_ID, GO_NORM_ID
from loctext.learning.evaluations import relation_accept_uniprot_go, GO_TREE
from nalaf import print_verbose, print_debug


#
# Note: GO:0005575 is the root of the cellular_component GO hierarchy
#


def test_relation_accept_uniprot_go_basic_eq():

    accept_prediction = relation_accept_uniprot_go

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|yyy",
        "r_5|n_7|xxx|n_8|yyy")

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0000123")

    assert accept_prediction(
        "r_5|n_7|P04637|n_8|yyy",
        "r_5|n_7|P04637|n_8|yyy")

    # Note, the following is a stub test relation and does not have to be biologically true
    assert accept_prediction(
        "r_5|n_7|P04637|n_8|GO:0000123",
        "r_5|n_7|P04637|n_8|GO:0000123")


def test_relation_accept_uniprot_go_basic_ne():

    accept_prediction = relation_accept_uniprot_go

    assert not accept_prediction(
        "r_5|n_7|xxx|n_8|yyy",
        "r_5|n_7|xxx_DIFFERENT|n_8|yyy")

    with raises(KeyError):
        assert not accept_prediction(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_7|xxx|n_8|yyy_DIFERENT")

    assert not accept_prediction(
        "r_5|n_7|xxx|n_8|yyy",
        "r_5|n_7|xxx_DIFFERENT|n_8|yyy_DIFERENT")


def test_relation_accept_uniprot_rel_type_is_not_compared():

    accept_prediction = relation_accept_uniprot_go

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|yyy",
        "r_DIFFERENT|n_7|xxx|n_8|yyy")


def test_relation_accept_uniprot_go_exceptions():

    accept_prediction = relation_accept_uniprot_go

    with raises(Exception) as puta:
        assert accept_prediction(
            "",
            "")

    with raises(Exception):
        assert accept_prediction(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_INVALID|xxx|n_8|yyy")

    with raises(Exception):
        assert accept_prediction(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_INVALID|xxx|n_8|yyy")

    with raises(Exception):
        assert accept_prediction(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_7|xxx|n_INVALID|yyy")

    with raises(Exception):
        assert accept_prediction(
            "r_5|n_INVALID|xxx|n_8|yyy",
            "r_5|n|xxx|n_8|yyy")

    with raises(Exception):
        assert accept_prediction(
            "r_5|n_7|xxx|n_INVALID|yyy",
            "r_5|n|xxx|n_8|yyy")


def test_relation_accept_uniprot_go_direct_children_ORDER_DOES_MATTER():
     # gold must be parecent to accept the prediction, not the other way around

    # see: http://www.ebi.ac.uk/QuickGO/GTerm?id=GO:0000123#term=ancchart

    # GO:0000123 (histone acetyltransferase complex) is a:
    # * direct child of GO:0044451 -- nucleoplasm part
    # * direct child of GO:0031248 -- protein acetyltransferase complex

    accept_prediction = relation_accept_uniprot_go

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0044451",
        "r_5|n_7|xxx|n_8|GO:0000123")

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0031248",
        "r_5|n_7|xxx|n_8|GO:0000123")

    # but...

    assert not accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0044451")

    assert not accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    # and not related at all with with one another as parent or child...

    assert not accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0031248",
        "r_5|n_7|xxx|n_8|GO:0044451")

    assert not accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0044451",
        "r_5|n_7|xxx|n_8|GO:0031248")


def test_relation_accept_uniprot_go_indirect_children():
    # see: http://www.ebi.ac.uk/QuickGO/GTerm?id=GO:0000123#term=ancchart

    # GO:0000123 (histone acetyltransferase complex) is a:
    # * direct child of GO:0044451 -- nucleoplasm part
    # * direct child of GO:0031248 -- protein acetyltransferase complex
    #
    # * indirect child of GO:0044428 (nuclear part) through GO:0044451
    # * indirect child of GO:0005634 (nucleus) through GO:0044451 --> GO:0044428
    #
    # * indirect child of GO:0005575 (cellular_component) since this is the root of the cellular component ontology

    accept_prediction = relation_accept_uniprot_go
    ignore_prediction = (lambda gold, pred: relation_accept_uniprot_go(gold, pred) is None)

    # Accept when the prediciton is more detailed than gold (that is, the prediction is an in-/direct child of gold)

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0044451",
        "r_5|n_7|xxx|n_8|GO:0000123")

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0031248",
        "r_5|n_7|xxx|n_8|GO:0000123")

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0031248",
        "r_5|n_7|xxx|n_8|GO:0000123")

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0031248",
        "r_5|n_7|xxx|n_8|GO:0000123")

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0005575",
        "r_5|n_7|xxx|n_8|GO:0000123")

    # Ignore when the prediciton is above gold (that is, the prediction is an in-/direct parent of gold)

    assert ignore_prediction(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0044451")

    assert ignore_prediction(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    assert ignore_prediction(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    assert ignore_prediction(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    assert ignore_prediction(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0005575")


def test_relation_accept_uniprot_go_all_children_of_root():
    # all go terms are indirect children of the root, cellular_component=GO:0005575, including the root itself
    # Therefore:
    #   1) if gold=root, all predictions are True (accept)
    #   2) if pred=root, all predictions are None (ignore)

    accept_prediction = relation_accept_uniprot_go
    ignore_prediction = (lambda gold, pred: relation_accept_uniprot_go(gold, pred) is None)

    for go_term in GO_TREE:
        pred_parents = GO_TREE[go_term].parents

        no_parent_in_ontology = all(p not in GO_TREE for p in pred_parents)
        is_root = len(pred_parents) == 0
        not_in_ontology = no_parent_in_ontology and not is_root

        # or not_in_ontology, go_term + " < " + ','.join(pred_parents)

        assert accept_prediction(
            "r_5|n_7|xxx|n_8|GO:0005575",
            "r_5|n_7|xxx|n_8|" + go_term), go_term + " < " + ','.join(pred_parents)

        assert ignore_prediction(
            "r_5|n_7|xxx|n_8|" + go_term,
            "r_5|n_7|xxx|n_8|GO:0005575"), go_term + " < " + ','.join(pred_parents)

    # The following tests the root with itself

    assert accept_prediction(
        "r_5|n_7|xxx|n_8|GO:0005575",
        "r_5|n_7|xxx|n_8|GO:0005575")

    # The following tests check that the root is appropriately handled without being an arbitrary/random/fake string

    # Note, here the gold fake go term IS checked and that's why the expected error
    with raises(KeyError):
        assert not accept_prediction(
            "r_5|n_7|xxx|n_8|GO:0005575",
            "r_5|n_7|xxx|n_8|FAKE")

    # Note, here the gold fake go term IS checked and that's why the expected error
    with raises(KeyError):
        assert not accept_prediction(
            "r_5|n_7|xxx|n_8|FAKE",
            "r_5|n_7|xxx|n_8|GO:0005575")


def test_relation_accept_uniprot_uniprots_as_list():

    accept_prediction = relation_accept_uniprot_go

    # Note, the following is a stub test relation and does not have to be biologically true
    assert accept_prediction(
        "r_5|n_7|P04637|n_8|yyy",
        "r_5|n_7|P04637,P02340|n_8|yyy")

    # Note, the following is stub test relation and does not have to be biologically true
    assert accept_prediction(
        "r_5|n_7|P04637|n_8|yyy",
        "r_5|n_7|P02340,P04637|n_8|yyy")

    # Note, the following is a stub test relation and does not have to be biologically true
    assert accept_prediction(
        "r_5|n_7|P04637|n_8|yyy",
        "r_5|n_7|in_the_middle:,P04637,P02340|n_8|yyy")

    # Note, the following is stub test relation and does not have to be biologically true
    assert accept_prediction(
        "r_5|n_7|P04637,P02340|n_8|yyy",
        "r_5|n_7|P04637|n_8|yyy")

    # Note, the following is stub test relation and does not have to be biologically true
    assert accept_prediction(
        "r_5|n_7|P02340,P04637|n_8|yyy",
        "r_5|n_7|P04637|n_8|yyy")

    # Note, the following is stub test relation and does not have to be biologically true
    assert accept_prediction(
        "r_5|n_7|P02340,P04637,:in_the_middle|n_8|yyy",
        "r_5|n_7|P04637|n_8|yyy")


def test_relation_accept_uniprot_uniprots_as_list_do_not_have_to_be_valid():

    accept_prediction = relation_accept_uniprot_go

    assert accept_prediction(
        "r_5|n_7|a|n_8|yyy",
        "r_5|n_7|a,b|n_8|yyy")

    assert accept_prediction(
        "r_5|n_7|a,b|n_8|yyy",
        "r_5|n_7|a|n_8|yyy")

    assert accept_prediction(
        "r_5|n_7| a ,b,,|n_8|yyy",
        "r_5|n_7| a |n_8|yyy")

    assert accept_prediction(
        "r_5|n_7| a |n_8|yyy",
        "r_5|n_7| a ,b,,|n_8|yyy")


if __name__ == "__main__":

    # selected tests:

    test_relation_accept_uniprot_go_direct_children_ORDER_DOES_MATTER()
    test_relation_accept_uniprot_go_all_children_of_root()
    test_relation_accept_uniprot_uniprots_as_list()
