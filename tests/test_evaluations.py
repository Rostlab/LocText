# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass

from pytest import raises
from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path, UNIPROT_NORM_ID, GO_NORM_ID
from loctext.learning.evaluations import relation_equals_uniprot_go, GO_TREE
from nalaf import print_verbose, print_debug


def test_relation_equals_uniprot_go_basic_eq():

    are_equivalent = relation_equals_uniprot_go

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|yyy",
        "r_5|n_7|xxx|n_8|yyy")

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0000123")

    assert are_equivalent(
        "r_5|n_7|P04637|n_8|yyy",
        "r_5|n_7|P04637|n_8|yyy")

    # Note, the following is a stub test relation and does not have to be biologically true
    assert are_equivalent(
        "r_5|n_7|P04637|n_8|GO:0000123",
        "r_5|n_7|P04637|n_8|GO:0000123")


def test_relation_equals_uniprot_go_basic_ne():

    are_equivalent = relation_equals_uniprot_go

    assert not are_equivalent(
        "r_5|n_7|xxx|n_8|yyy",
        "r_5|n_7|xxx_DIFFERENT|n_8|yyy")

    assert not are_equivalent(
        "r_5|n_7|xxx|n_8|yyy",
        "r_5|n_7|xxx|n_8|yyy_DIFERENT")

    assert not are_equivalent(
        "r_5|n_7|xxx|n_8|yyy",
        "r_5|n_7|xxx_DIFFERENT|n_8|yyy_DIFERENT")

    assert not are_equivalent(
        "r_5|n_7|xxx|n_8|yyy",
        "r_DIFFERENT|n_7|xxx|n_8|yyy")


def test_relation_equals_uniprot_go_exceptions():

    are_equivalent = relation_equals_uniprot_go

    with raises(Exception) as puta:
        assert are_equivalent(
            "",
            "")

    with raises(Exception):
        assert are_equivalent(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_INVALID|xxx|n_8|yyy")

    with raises(Exception):
        assert are_equivalent(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_INVALID|xxx|n_8|yyy")

    with raises(Exception):
        assert are_equivalent(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_7|xxx|n_INVALID|yyy")

    with raises(Exception):
        assert are_equivalent(
            "r_5|n_INVALID|xxx|n_8|yyy",
            "r_5|n|xxx|n_8|yyy")

    with raises(Exception):
        assert are_equivalent(
            "r_5|n_7|xxx|n_INVALID|yyy",
            "r_5|n|xxx|n_8|yyy")


def test_relation_equals_uniprot_go_ORDER_DOES_MATTER():
    # see: http://www.ebi.ac.uk/QuickGO/GTerm?id=GO:0000123#term=ancchart

    # GO:0000123 is a:
    # * direct child of GO:0044451 -- nucleoplasm part
    # * direct child of GO:0031248 -- protein acetyltransferase complex

    are_equivalent = relation_equals_uniprot_go

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0044451")

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    # but...

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0044451",
        "r_5|n_7|xxx|n_8|GO:0000123")

    assert not are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0031248",
        "r_5|n_7|xxx|n_8|GO:0000123")


def test_relation_equals_uniprot_go_indirect_children():
    # see: http://www.ebi.ac.uk/QuickGO/GTerm?id=GO:0000123#term=ancchart

    # GO:0000123 (histone acetyltransferase complex) is a:
    # * direct child of GO:0044451 -- nucleoplasm part
    # * direct child of GO:0031248 -- protein acetyltransferase complex
    #
    # * indirect child of GO:0044428 (nuclear part) through GO:0044451
    # * indirect child of GO:0005634 (nucleus) through GO:0044451 --> GO:0044428
    #
    # * indirect child of GO:0005575 (cellular_component) since this is the root of the cellular component ontology

    are_equivalent = relation_equals_uniprot_go

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0044451")

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0005575")


def test_relation_equals_uniprot_go_all_children_of_root():
    # all go terms are indirect children of the root (including the root itself)

    are_equivalent = relation_equals_uniprot_go

    for go_term in GO_TREE:
        assert are_equivalent(
            go_term,
            "r_5|n_7|xxx|n_8|GO:0005575")


def test_relation_equals_uniprot_uniprots_as_list():

    are_equivalent = relation_equals_uniprot_go

    # Note, the following is a stub test relation and does not have to be biologically true
    assert are_equivalent(
        "r_5|n_7|P04637|n_8|GO:0000123",
        "r_5|n_7|P04637,P02340|n_8|GO:0000123")

    # Note, the following is stub test relation and does not have to be biologically true
    assert are_equivalent(
        "r_5|n_7|P04637|n_8|GO:0000123",
        "r_5|n_7|P02340,P04637|n_8|GO:0000123")

    # Note, the following is stub test relation and does not have to be biologically true
    assert are_equivalent(
        "r_5|n_7|P04637,P02340|n_8|GO:0000123",
        "r_5|n_7|P04637|n_8|GO:0000123")

    # Note, the following is stub test relation and does not have to be biologically true
    assert are_equivalent(
        "r_5|n_7|P02340,P04637|n_8|GO:0000123",
        "r_5|n_7|P04637|n_8|GO:0000123")


if __name__ == "__main__":

    test_relation_equals_uniprot_go()
