from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.utils.graph import get_path, build_walks
from nalaf import print_debug


def combine_sentences(sentence1, sentence2):
    """
    Combine two simple simple normal sentences into a "chained" sentence with
    dependecies and paths created as necessary for the DS model.

    `createCombinedSentence` re-implementation of Shrikant's (java) into Python.

    Each sentence is a list of Tokens as defined in class Part (nalaf: data.py).

    The sentences are assumed, but not asserted, to be different and sorted:
    sentence1 must be before sentence2.
    """

    combined_sentence = sentence1 + sentence2

    combined_sentence = _add_extra_links(combined_sentence, sentence1, sentence2)

    return combined_sentence


def get_root_token(sentence, feature_is_root='is_root'):
    """
    See parsers.py :: SpacyParser.
    """
    roots = [token for token in sentence if token.features[feature_is_root] is True]
    assert len(roots) == 1, "The sentence contains {} roots (?). Expected: 1 -- Sentence: {}".format(len(roots), ' '.join(sentence))

    return roots[0]


def _add_extra_links(combined_sentence, sentence1, sentence2):
    """
    `addExtraLinks` re-implementation of Shrikant's (java) into Python.

    Some comments and commented-out code exactly as original java code.
    """

    # TODO addWordSimilarityLinks(combSentence, tokenOffset)

    # TODO addProteinLinks(combSentence, tokenOffset)

    # Just as we added the links from "protein" to actual protein entities
    # add the links from "location"/"localization" to location entity
    # TODO addLocationLinks(combSentence, tokenOffset)

    # addProteinFamilyLinks(combSentence, tokenOffset);

    _addRootLinks(combined_sentence, sentence1, sentence2)

    # addShortFormLinks(combSentence, prevSentence, currSentence)

    return combined_sentence


def _addRootLinks(combined_sentence, sentence1, sentence2):
    """
    link roots of both the sentences

    Dependency directions:

    sentence1 -> sentence2
    sentence2 <- sentence1
    """

    root_sent_1 = get_root_token(sentence1)
    root_sent_2 = get_root_token(sentence2)

    root_sent_1.features['dependency_to'] = (root_sent_2, "rootDepForward")
    root_sent_1.features['dependency_from'] = (root_sent_2, "rootDepBackward")

    root_sent_2.features['dependency_from'] = (root_sent_1, "rootDepForward")
    root_sent_2.features['dependency_to'] = (root_sent_1, "rootDepBackward")

    return combined_sentence


class BigramFeatureGenerator(EdgeFeatureGenerator):
    """
    `buildBigramFeatures` re-implementation of Shrikant's (java) into Python.
    """

    def __init__(
    self,
    prefix_dependency_from_prot_entity_to_prot_word=None,
    prefix_dependency_from_prot_word_to_prot_entity=None,
    prefix_PWPE_bow=None,
    prefix_PWPE_pos=None,
    prefix_PWPE_bow_masked=None,
    prefix_PWPE_dep=None,
    prefix_PWPE_dep_full=None,
    prefix_PWPE_dep_gram=None,
    prefix_protein_word_found=None,
    prefix_protein_not_word_found=None
    ):

    def generate(dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            (sentence1, sentence2) = edge.get_sentences_pair(force_sort=True)

            head1 = edge.entity1.head_token
            head2 = edge.entity2.head_token


    def generate(combined_sentence, feature_set, is_training_mode):

        for(int i=0; i<sentence.getTokenList().size()-1; i++):
            Token currToken = sentence.getTokenList().get(i);
            Token nextToken = sentence.getTokenList().get(i+1);

            String currTokenText = currToken.getTokenText();
            String nextTokenText = nextToken.getTokenText();

            String textFeature = "bow_" + currTokenText + "_" + nextTokenText;
            addToFeatureSet("5_" + textFeature, 1, curEdgeFeatureSet);

            String cTokTextMask = currToken.getTokenTextMasked();
            String nTokTextMask = nextToken.getTokenTextMasked();

            String textMaskedFeature = "bowMasked_" + cTokTextMask + "_" + nTokTextMask;
            addToFeatureSet("6_" + textMaskedFeature, 1, curEdgeFeatureSet);

            String currTokenPOS = currToken.getPOSTag();
            String nextTokenPOS = nextToken.getPOSTag();

            String posFeature = "pos_" + currTokenPOS + "_" + nextTokenPOS;
            addToFeatureSet("7_" + posFeature, 1, curEdgeFeatureSet);

            String currTokStem = currToken.getStem();
            String nextTokStem = nextToken.getStem();

            String stemFeature = "stem_" + currTokStem + "_" + nextTokStem;
            addToFeatureSet("8_" + stemFeature, 1, curEdgeFeatureSet);
            }
