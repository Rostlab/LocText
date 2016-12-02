from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.utils.graph import get_path, build_walks
from nalaf import print_debug
from nltk.stem import PorterStemmer


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
    assert len(roots) == 1, "The sentence contains {} roots (?). Expected: 1 -- Sentence: {}".format(len(roots), ' '.join((t.word for t in sentence)))

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
        prefix_bow=None,
        prefix_masked=None,
        prefix_pos=None,
        prefix_stem=None
    ):

        self.stemmer = PorterStemmer()
        """an instance of the PorterStemmer"""

        self.prefix_bow = prefix_bow
        self.prefix_masked = prefix_masked
        self.prefix_pos = prefix_pos
        self.prefix_stem = prefix_stem


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            (sentence1, sentence2) = edge.get_sentences_pair(force_sort=True)
            combined_sentence = combine_sentences(sentence1, sentence2)

            # head1 = edge.entity1.head_token
            # head2 = edge.entity2.head_token

            self._generate(feature_set, is_training_mode, edge, combined_sentence)


    def _generate(self, feature_set, is_training_mode, edge, combined_sentence):

        for currToken, nextToken, in zip(combined_sentence, combined_sentence[1:]):
            pass

            # # TODO should it be lowercase ???
            # feature_name = self.gen_prefix_feat_name("prefix_bow", currToken.word, nextToken.word)
            # self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            #
            # String textFeature = "bow_" + currTokenText + "_" + nextTokenText;
            # addToFeatureSet("5_" + textFeature, 1, curEdgeFeatureSet);
            #
            # String cTokTextMask = currToken.getTokenTextMasked();
            # String nTokTextMask = nextToken.getTokenTextMasked();
            #
            # String textMaskedFeature = "bowMasked_" + cTokTextMask + "_" + nTokTextMask;
            # addToFeatureSet("6_" + textMaskedFeature, 1, curEdgeFeatureSet);
            #
            # String currTokenPOS = currToken.getPOSTag();
            # String nextTokenPOS = nextToken.getPOSTag();
            #
            # String posFeature = "pos_" + currTokenPOS + "_" + nextTokenPOS;
            # addToFeatureSet("7_" + posFeature, 1, curEdgeFeatureSet);
            #
            # String currTokStem = currToken.getStem();
            # String nextTokStem = nextToken.getStem();
            #
            # String stemFeature = "stem_" + currTokStem + "_" + nextTokStem;
            # addToFeatureSet("8_" + stemFeature, 1, curEdgeFeatureSet);
