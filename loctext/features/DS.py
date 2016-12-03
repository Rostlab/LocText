from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.utils.graph import get_path, build_walks
from nalaf import print_debug
from nltk.stem import PorterStemmer
from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID


def combine_sentences(edge, sentence1, sentence2):
    """
    Combine two simple simple normal sentences into a "chained" sentence with
    dependecies and paths created as necessary for the DS model.

    `createCombinedSentence` re-implementation of Shrikant's (java) into Python.

    Each sentence is a list of Tokens as defined in class Part (nalaf: data.py).

    The sentences are assumed, but not asserted, to be different and sorted:
    sentence1 must be before sentence2.
    """

    combined_sentence = sentence1 + sentence2

    combined_sentence = _add_extra_links(edge, combined_sentence, sentence1, sentence2)

    return combined_sentence


def get_sentence_roots(sentence, feature_is_root='is_root'):
    """
    See parsers.py :: SpacyParser.
    """
    roots = [token for token in sentence if token.features[feature_is_root] is True]
    assert len(roots) >= 1, "The sentence contains {} roots (?). Expected: >= 1 -- Sentence: {}".format(len(roots), ' '.join((t.word for t in sentence)))

    return roots


def _add_extra_links(edge, combined_sentence, sentence1, sentence2):
    """
    `addExtraLinks` re-implementation of Shrikant's (java) into Python.

    Some comments and commented-out code exactly as original java code.
    """

    _addWordSimilarityLinks(combined_sentence, sentence1, sentence2)

    # TODO would be better to not use the constants PRO_ID (protRef) and LOC_ID (locRef) (below) here -- It's hardcoded
    _addEntityLinks(edge, combined_sentence, sentence1, sentence2, PRO_ID, ['protein'], "protRef")

    # Just as we added the links from "protein" to actual protein entities
    # add the links from "location"/"localization" to location entity
    _addEntityLinks(edge, combined_sentence, sentence1, sentence2, LOC_ID, ['location', 'localiz', 'compartment'], "locRef")

    # addProteinFamilyLinks(combSentence, tokenOffset);

    _addRootLinks(combined_sentence, sentence1, sentence2)

    # addShortFormLinks(combSentence, prevSentence, currSentence)

    return combined_sentence


def _addWordSimilarityLinks(combined_sentence, sentence1, sentence2):
    """
    add links between words (nouns) that have same tokenText in prev & currSentence
    """
    from itertools import product

    for (s1_token, s2_token) in product(sentence1, sentence2):

        # NN, NNS, NNP, NNPS : https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        if "NN" == s1_token.features['pos'][0:2] == s2_token.features['pos'][0:2]:
            if s1_token.word == s2_token.word:
                s1_token.features['user_dependency_to'].append((s2_token, "wordSim"))
                s2_token.features['user_dependency_from'].append((s1_token, "wordSim"))

            # TODO note, here I'm using the (spacy) lemma, not the (Porter) stem
            if s1_token.features['lemma'] == s2_token.features['lemma']:
                s1_token.features['user_dependency_to'].append((s2_token, "stemSim"))
                s2_token.features['user_dependency_from'].append((s1_token, "stemSim"))


def _addEntityLinks(edge, combined_sentence, sentence1, sentence2, class_id, key_words, dependency_type):
    """
    `addProteinLinks` and `addLocationLinks` re-implementation of Shrikant's (java) into Python.
    """

    assert edge.same_part

    def _do_one_direction(sent_a, sent_b):

        sent_a_contains_entity = edge.entity1.class_id == class_id
        if not sent_a_contains_entity:
            sent_a_tokens_that_match_key_words = (t for t in sent_a if any(kw in t.word.lower() for kw in key_words))

            for sent_a_token in sent_a_tokens_that_match_key_words:

                for sent_b_token in sent_b:

                    sent_b_token_in_entity = sent_b_token.get_entity(edge.same_part)
                    if sent_b_token_in_entity is not None and sent_b_token_in_entity.class_id == class_id:

                        sent_a_token.features['user_dependency_to'].append((sent_b_token, dependency_type))
                        sent_b_token.features['user_dependency_from'].append((sent_a_token, dependency_type))

    # In combination: do both directions
    _do_one_direction(sentence1, sentence2)
    _do_one_direction(sentence2, sentence1)


def _addRootLinks(combined_sentence, sentence1, sentence2):
    """
    link roots of both the sentences

    `addRootLinks` re-implementation of Shrikant's (java) into Python.


    *IMPORTANT*:

    * Shrikant/Java/CoreNLP code had one single root for every sentence
    * Python/spaCy sentences can have more than 1 root
    * --> Therefore, we create a product of links of all the roots
    * --> see: (https://github.com/juanmirocks/LocText/issues/6#issue-177139892)


    Dependency directions:

    sentence1 -> sentence2
    sentence2 <- sentence1
    """
    from itertools import product

    for (s1_root, s2_root) in product(get_sentence_roots(sentence1), get_sentence_roots(sentence2)):

        s1_root.features['user_dependency_to'].append((s2_root, "rootDepForward"))
        s1_root.features['user_dependency_from'].append((s2_root, "rootDepBackward"))

        s2_root.features['user_dependency_from'].append((s1_root, "rootDepForward"))
        s2_root.features['user_dependency_to'].append((s1_root, "rootDepBackward"))


class BigramFeatureGenerator(EdgeFeatureGenerator):
    """
    `buildBigramFeatures` re-implementation of Shrikant's (java) into Python.
    """

    def __init__(
        self,
        prefix_bow=None,
        prefix_bow_masked=None,
        prefix_pos=None,
        prefix_stem=None
    ):

        self.stemmer = PorterStemmer()
        """an instance of the PorterStemmer"""

        self.prefix_bow = prefix_bow
        self.prefix_bow_masked = prefix_bow_masked
        self.prefix_pos = prefix_pos
        self.prefix_stem = prefix_stem


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            (sentence1, sentence2) = edge.get_sentences_pair(force_sort=True)
            combined_sentence = combine_sentences(edge, sentence1, sentence2)

            # head1 = edge.entity1.head_token
            # head2 = edge.entity2.head_token

            self._generate(feature_set, is_training_mode, edge, combined_sentence)


    def _generate(self, feature_set, is_training_mode, edge, combined_sentence):

        for currToken, nextToken, in zip(combined_sentence, combined_sentence[1:]):

            # TODO should it be lowercase ???
            feature_name = self.gen_prefix_feat_name("prefix_bow", currToken.word, nextToken.word)
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            feature_name = self.gen_prefix_feat_name("prefix_bow_masked", currToken.masked_text(edge.same_part), nextToken.masked_text(edge.same_part))
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            feature_name = self.gen_prefix_feat_name("prefix_pos", currToken.features['pos'], nextToken.features['pos'])
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            feature_name = self.gen_prefix_feat_name("prefix_stem", self.stemmer.stem(currToken.word), self.stemmer.stem(nextToken.word))
            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


class AnyNGramFeatureGenerator(EdgeFeatureGenerator):
    """
    `buildBigramFeatures` and `buildTrigramFeature` all-in-one re-implementation of Shrikant's (java) into Python.
    """

    def __init__(
        self,
        n_gram,
        prefix_bow=None,
        prefix_bow_masked=None,
        prefix_pos=None,
        prefix_stem=None
    ):

        self.n_gram = n_gram

        self.prefix_bow = prefix_bow
        self.prefix_bow_masked = prefix_bow_masked
        self.prefix_pos = prefix_pos
        self.prefix_stem = prefix_stem

        self.stemmer = PorterStemmer()
        """an instance of the PorterStemmer"""

        self.features = [
            ("prefix_bow", lambda tok, _: tok.word),  # TODO should it be lowercase ???
            ("prefix_bow_masked", lambda tok, edge: tok.masked_text(edge.same_part)),
            ("prefix_pos", lambda tok, _: tok.features['pos']),
            ("prefix_stem", lambda tok, _: self.stemmer.stem(tok.word))
        ]


    def generate(self, dataset, feature_set, is_training_mode):

        for edge in dataset.edges():
            (sentence1, sentence2) = edge.get_sentences_pair(force_sort=True)
            combined_sentence = combine_sentences(edge, sentence1, sentence2)

            n_grams = zip(*(combined_sentence[start:] for start in range(0, self.n_gram)))

            for tokens in n_grams:

                for feature_pair in self.features:
                    transformed_tokens = (feature_pair[1](t, edge) for t in tokens)

                    feature_name = self.gen_prefix_feat_name(feature_pair[0], *transformed_tokens)
                    self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
