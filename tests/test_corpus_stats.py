# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError: # Parent module '' not loaded, cannot perform relative import
    pass

from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.learning.taggers import StubSameSentenceRelationExtractor, StubRelationExtractor
from loctext.learning.train import read_corpus, evaluate_with_argv
from nalaf import print_verbose, print_debug
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator
import math
import sys
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf.preprocessing.tokenizers import NLTK_TOKENIZER
from nalaf.structures.data import Entity
from collections import Counter


def test_count_relations_dists_texts_with_repetitions():
    entity_map_fun = Entity.__repr__

    # Documents 100
    nums = Counter({'D0': 351, 'D1': 95, 'D2': 53, 'D3': 23, 'D5': 9, 'D6': 8, 'D4': 7, 'D7': 2, 'D9': 2})
    percts = Counter({'D0': 0.6381818181818182, 'D1': 0.17272727272727273, 'D2': 0.09636363636363636, 'D3': 0.04181818181818182, 'D5': 0.016363636363636365, 'D6': 0.014545454545454545, 'D4': 0.012727272727272728, 'D9': 0.0036363636363636364, 'D7': 0.0036363636363636364})

    sum_0_1 = 0.81  # D0 + D1 (just as Shrikant originally indicated)

    #

    corpus = read_corpus("LocText")

    sentence_splitter = NLTKSplitter()
    tokenizer = NLTK_TOKENIZER
    sentence_splitter.split(corpus)
    tokenizer.tokenize(corpus)

    print("# Documents", len(corpus))
    (counter_texts_nums, counter_texts_percts) = corpus.compute_stats_relations_distances(REL_PRO_LOC_ID, entity_map_fun)
    print(counter_texts_nums)
    print(counter_texts_percts)

    assert math.isclose(sum_0_1, (counter_texts_percts['D0'] + counter_texts_percts['D1']), abs_tol=0.01)


def test_count_relations_dists_texts_without_repetitions():
    entity_map_fun = DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['lowercased']

    # Documents 100
    nums = Counter({'D0': 292, 'D1': 78, 'D2': 48, 'D3': 22, 'D5': 9, 'D6': 7, 'D4': 7, 'D9': 2, 'D7': 1})
    percts = Counter({'D0': 0.6266094420600858, 'D1': 0.16738197424892703, 'D2': 0.10300429184549356, 'D3': 0.04721030042918455, 'D5': 0.019313304721030045, 'D6': 0.015021459227467811, 'D4': 0.015021459227467811, 'D9': 0.004291845493562232, 'D7': 0.002145922746781116})

    sum_0_1 = 0.79  # D0 + D1

    #

    corpus = read_corpus("LocText")

    sentence_splitter = NLTKSplitter()
    tokenizer = NLTK_TOKENIZER
    sentence_splitter.split(corpus)
    tokenizer.tokenize(corpus)

    print("# Documents", len(corpus))
    (counter_texts_nums, counter_texts_percts) = corpus.compute_stats_relations_distances(REL_PRO_LOC_ID, entity_map_fun)
    print(counter_texts_nums)
    print(counter_texts_percts)

    assert math.isclose(sum_0_1, (counter_texts_percts['D0'] + counter_texts_percts['D1']), abs_tol=0.01)


def test_count_relations_dists_normalizations_without_repetitions():
    entity_map_fun = DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_first']

    # Documents 100

    # Texts With Repetitions
    Counter({'D0': 216, 'D1': 57, 'D2': 32, 'D3': 15, 'D5': 9, 'D6': 6, 'D4': 5, 'D9': 2})
    Counter({'D0': 0.631578947368421, 'D1': 0.16666666666666666, 'D2': 0.0935672514619883, 'D3': 0.043859649122807015, 'D5': 0.02631578947368421, 'D6': 0.017543859649122806, 'D4': 0.014619883040935672, 'D9': 0.005847953216374269})

    sum_0_1 = 0.80  # D0 + D1

    corpus = read_corpus("LocText")

    sentence_splitter = NLTKSplitter()
    tokenizer = NLTK_TOKENIZER
    sentence_splitter.split(corpus)
    tokenizer.tokenize(corpus)

    print("# Documents", len(corpus))
    (counter_texts_nums, counter_texts_percts) = corpus.compute_stats_relations_distances(REL_PRO_LOC_ID, entity_map_fun)
    print(counter_texts_nums)
    print(counter_texts_percts)

    # for e in corpus.entities():
    #     print(e, e.normalisation_dict, is_empty)

    assert math.isclose(sum_0_1, (counter_texts_percts['D0'] + counter_texts_percts['D1']), abs_tol=0.01)
