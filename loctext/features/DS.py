from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.utils.graph import get_path, build_walks
from nalaf import print_debug


def buildBigramFeatures


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
            (sentence1, sentence2) = edge.get_sentences_pair()

            head1 = edge.entity1.head_token
            head2 = edge.entity2.head_token


    def generate(Sentence sentence, feature_set, is_training_mode):

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
