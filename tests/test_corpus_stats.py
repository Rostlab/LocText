# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass

from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.learning.taggers import StubSamePartRelationExtractor, StubRelationExtractor
from loctext.learning.train import read_corpus, evaluate_with_argv
from nalaf import print_verbose, print_debug
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator
import math
import sys
from nalaf.structures.data import Entity
from collections import Counter
from loctext.learning.evaluations import relation_accept_uniprot_go, GO_TREE


CORPUS_PERCENTANGE = 1.0


def test_count_relations_dists_with_repetitions():
    _test(
        Entity.__repr__,
        None,  # meaning: str.__eq__
        #
        0.81,
        #
        Counter({'D0': 351, 'D1': 95, 'D2': 53, 'D3': 23, 'D5': 9, 'D6': 8, 'D4': 7, 'D9': 2, 'D7': 2}),
        Counter({'D0': 0.6381818181818182, 'D1': 0.17272727272727273, 'D2': 0.09636363636363636, 'D3': 0.04181818181818182, 'D5': 0.016363636363636365, 'D6': 0.014545454545454545, 'D4': 0.012727272727272728, 'D9': 0.0036363636363636364, 'D7': 0.0036363636363636364})
    )

def test_count_relations_dists_without_repetitions():
    _test(
        DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['lowercased'],
        None,  # meaning: str.__eq__
        #
        0.80,
        #
        Counter({'D0': 278, 'D1': 78, 'D2': 44, 'D3': 22, 'D5': 9, 'D6': 7, 'D4': 5, 'D9': 2, 'D7': 1}),
        Counter({'D0': 0.6233183856502242, 'D1': 0.17488789237668162, 'D2': 0.09865470852017937, 'D3': 0.04932735426008968, 'D5': 0.020179372197309416, 'D6': 0.01569506726457399, 'D4': 0.011210762331838564, 'D9': 0.004484304932735426, 'D7': 0.002242152466367713})
    )


def test_count_relations_dists_normalizations_without_repetitions():
    _test(
        DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_first'],
        None,  # meaning: str.__eq__
        #
        0.81,
        #
        Counter({'D0': 215, 'D1': 54, 'D2': 31, 'D3': 12, 'D5': 9, 'D6': 6, 'D9': 2, 'D4': 2}),
        Counter({'D0': 0.649546827794562, 'D1': 0.16314199395770393, 'D2': 0.09365558912386707, 'D3': 0.03625377643504532, 'D5': 0.027190332326283987, 'D6': 0.01812688821752266, 'D9': 0.006042296072507553, 'D4': 0.006042296072507553})
    )


def test_count_relations_dists_normalizations_without_repetitions_considering_hierarchy():
    _test(
        DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_first'],
        relation_accept_uniprot_go,
        #
        0.83,
        #
        Counter({'D0': 179, 'D1': 46, 'D2': 21, 'D3': 10, 'D5': 7, 'D6': 5, 'D9': 2, 'D4': 2}),
        Counter({'D0': 0.6580882352941176, 'D1': 0.16911764705882354, 'D2': 0.07720588235294118, 'D3': 0.03676470588235294, 'D5': 0.025735294117647058, 'D6': 0.01838235294117647, 'D9': 0.007352941176470588, 'D4': 0.007352941176470588})
    )


#


def _test(entity_map_fun, relation_accept_fun, expected_sum_perct_d0_d1, expected_nums, expected_percts):
    corpus = read_corpus("LocText", CORPUS_PERCENTANGE)

    # Note: the predictor will already split & tokenize the corpus. See the implementation for details
    StubSamePartRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID).annotate(corpus)

    (counter_nums, counter_percts) = corpus.compute_stats_relations_distances(REL_PRO_LOC_ID, entity_map_fun, relation_accept_fun, predicted=True)

    print()
    print("# Documents", len(corpus))
    print("# Relations", sum(counter_nums.values()))
    print("  ", counter_nums)
    print("  ", counter_percts)

    assert expected_nums == counter_nums
    assert math.isclose(expected_sum_perct_d0_d1, (counter_percts['D0'] + counter_percts['D1']), abs_tol=0.01)
