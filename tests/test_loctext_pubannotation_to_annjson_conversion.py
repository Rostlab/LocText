# Be able to call directly such as `python test_annotators.py`
try:
    from .context import loctext
except SystemError:  # Parent module '' not loaded, cannot perform relative import
    pass

from pytest import raises
from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path, UNIPROT_NORM_ID, GO_NORM_ID
from loctext.learning.evaluations import relation_equals_uniprot_go, GO_TREE
from nalaf import print_verbose, print_debug
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf.learning.taggers import StubRelationExtractor
from loctext.learning.train import read_corpus, evaluate_with_argv
from nalaf import print_verbose, print_debug
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator
import math
import sys


def test_same_stats():

    original = read_corpus("LocText_original", corpus_percentage=1.0)
    newone = read_corpus("LocText", corpus_percentage=1.0)

    # Verification
    original.validate_entity_offsets()
    newone.validate_entity_offsets()

    # Basic
    assert 100 == len(original) == len(newone)
    assert len(list(original.entities())) == len(list(newone.entities())) and len(list(original.entities())) > 0
    assert 0 == len(list(original.predicted_entities())) == len(list(newone.predicted_entities()))
    assert len(list(original.relations())) == len(list(newone.relations())) and len(list(original.relations())) > 0
    assert 0 == len(list(original.predicted_relations())) == len(list(newone.predicted_relations()))

    # Elaborated
    edge_generator_d0 = SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=0)
    annotator = StubRelationExtractor(edge_generator_d0)

    annotator.annotate(original)
    annotator.annotate(newone)

    assert len(list(original.edges())) > 0 and (len(list(original.edges())) == len(list(newone.edges())) == len(list(newone.predicted_relations())))
    num_d0 = len(list(newone.predicted_relations()))

    edge_generator_d1 = SentenceDistanceEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID, distance=1)
    annotator = StubRelationExtractor(edge_generator_d1)

    annotator.annotate(original)
    annotator.annotate(newone)

    assert len(list(original.edges())) > 0 and (len(list(original.edges())) == len(list(newone.edges())) == (- num_d0 + len(list(newone.predicted_relations()))))

    # Normalizations
    assert all(len(e.normalisation_dict) == 0 for e in original.entities())

    count_normalizations = 0

    for e in newone.entities():
        if str(e.class_id) != "e_4":
            assert len(e.normalisation_dict) == 1, e
            norm_id = next(iter(e.normalisation_dict.values()))
            assert type(norm_id) is str, e   # do not write arrays, only comma-separated strings
            assert ' ' not in norm_id, e   # We cannot have stuff like 'GO:0005811 lipid droplet' -- let's have only the GO id

        assert not e.normalisation_dict or e.normalisation_dict

        count_normalizations += 1

    assert count_normalizations == len(list(newone.entities())) == len(list(original.entities()))


if __name__ == "__main__":

    # selected tests:

    test_same_stats()
