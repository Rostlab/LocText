# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass

import pytest
from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path, UNIPROT_NORM_ID, GO_NORM_ID
from loctext.learning.evaluations import relation_equals_uniprot_go
from nalaf import print_verbose, print_debug


def test_relation_equals_uniprot_go_basic_eq():

    are_equivalent = relation_equals_uniprot_go

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|yyy",
        "r_5|n_7|xxx|n_8|yyy")


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

    with pytest.raises(Exception) as puta:
        assert are_equivalent(
            "",
            "")

    with pytest.raises(Exception):
        assert are_equivalent(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_INVALID|xxx|n_8|yyy")

        assert are_equivalent(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_INVALID|xxx|n_8|yyy")

        assert are_equivalent(
            "r_5|n_7|xxx|n_8|yyy",
            "r_5|n_7|xxx|n_INVALID|yyy")

        assert are_equivalent(
            "r_5|n_INVALID|xxx|n_8|yyy",
            "r_5|n|xxx|n_8|yyy")

        assert are_equivalent(
            "r_5|n_7|xxx|n_INVALID|yyy",
            "r_5|n|xxx|n_8|yyy")


def test_relation_equals_uniprot_go_ORDER_DOES_MATTER():
    # Note: GO:0000123 is a direct child of GO:0031248 -- see: http://www.ebi.ac.uk/QuickGO/GTerm?id=GO:0000123#term=ancchart

    are_equivalent = relation_equals_uniprot_go

    assert are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0000123",
        "r_5|n_7|xxx|n_8|GO:0031248")

    assert not are_equivalent(
        "r_5|n_7|xxx|n_8|GO:0031248",
        "r_5|n_7|xxx|n_8|GO:0000123")


if __name__ == "__main__":

    test_relation_equals_uniprot_go()
