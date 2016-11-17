# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass

from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path, UNIPROT_NORM_ID, GO_NORM_ID
from loctext.learning.evaluations import relation_equals_uniprot_go
from nalaf import print_verbose, print_debug


def test_relation_equals_uniprot_go():
    assert False


if __name__ == "__main__":

    test_relation_equals_uniprot_go()
