from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.utils.graph import get_path, build_walks
from nalaf import print_debug


class ProteinWordFeatureGenerator(EdgeFeatureGenerator):
    """
    Check for the presence of the word "protein" in the sentence. If the word
    "protein" is part of an entity, then it checks for dependencies from the
    head token of the entity to the word and vice versa.

    For the dependency path between the word "protein" and the head token, it
    also calculates the bag of words representation, masked text and parts of
    speech for each token in the path.

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param graphs: the graph representation for each sentence in the dataset
    :type graphs: dictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(
        self, graphs,
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
        self.graphs = graphs
        """a dictionary of graphs to avoid recomputation of path"""

        self.prefix_dependency_from_prot_entity_to_prot_word = prefix_dependency_from_prot_entity_to_prot_word
        self.prefix_dependency_from_prot_word_to_prot_entity = prefix_dependency_from_prot_word_to_prot_entity
        self.prefix_PWPE_bow = prefix_PWPE_bow
        self.prefix_PWPE_pos = prefix_PWPE_pos
        self.prefix_PWPE_bow_masked = prefix_PWPE_bow_masked
        self.prefix_PWPE_dep = prefix_PWPE_dep
        self.prefix_PWPE_dep_full = prefix_PWPE_dep_full
        self.prefix_PWPE_dep_gram = prefix_PWPE_dep_gram
        self.prefix_protein_word_found = prefix_protein_word_found
        self.prefix_protein_not_word_found = prefix_protein_not_word_found

        self.keyword = 'protein'


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            head1 = edge.entity1.head_token
            # head2 = edge.entity2.head_token
            sentence = edge.part.sentences[edge.sentence_id]
            protein_word_found = False
            for token in sentence:
                if token.is_entity_part(edge.part) and token.word.lower().find(self.keyword) >= 0:
                    protein_word_found = True
                    token_from = token.features['dependency_from'][0]

                    if token_from == head1:
                        feature_name = self.gen_prefix_feat_name("prefix_dependency_from_prot_entity_to_prot_word")
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                    for dependency_to in token.features['dependency_to']:
                        token_to = dependency_to[0]
                        if token_to == head1:
                            feature_name = self.gen_prefix_feat_name("prefix_dependency_from_prot_word_to_prot_entity")
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                        path = get_path(token, head1, edge.part, edge.sentence_id, self.graphs)
                        if path == []:
                            path = [token, head1]
                        for tok in path:
                            feature_name = self.gen_prefix_feat_name("prefix_PWPE_bow", tok.word)
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                            feature_name = self.gen_prefix_feat_name("prefix_PWPE_pos", tok.features['pos'])
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                            feature_name = self.gen_prefix_feat_name("prefix_PWPE_bow_masked", tok.masked_text(edge.part))
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                        all_walks = build_walks(path)
                        for dep_list in all_walks:
                            dep_path = ''
                            for dep in dep_list:
                                feature_name = self.gen_prefix_feat_name("prefix_PWPE_dep", dep[1])
                                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
                                dep_path += dep[1]

                            feature_name = self.gen_prefix_feat_name("prefix_PWPE_dep_full", dep_path)
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                        for j in range(len(all_walks)):
                            dir_grams = ''
                            for i in range(len(path) - 1):
                                cur_walk = all_walks[j]
                                if cur_walk[i][0] == path[i]:
                                    dir_grams += 'F'
                                else:
                                    dir_grams += 'R'

                            feature_name = self.gen_prefix_feat_name("prefix_PWPE_dep_gram", dir_grams)
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            if protein_word_found:
                feature_name = self.gen_prefix_feat_name("prefix_protein_word_found")
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            else:
                feature_name = self.gen_prefix_feat_name("prefix_protein_not_word_found")
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)


class LocationWordFeatureGenerator(EdgeFeatureGenerator):
    """
    Check each sentence for the presence of location words if the sentence
    contains an edge. These location words include ['location', 'localize'].

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, loc_e_id, prefix1, prefix2=None, prefix3=None):
        self.loc_e_id = loc_e_id
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.prefix3 = prefix3

        self.loc_tokens = [
            # Original LocText code
            'location',
            'localize',
            # Extra Added by Juanmi
            'localization',
        ]


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            location_word = False

            if edge.entity1.class_id != self.loc_e_id:  # 'e_1' (protein)
                head1 = edge.entity1.head_token
                head2 = edge.entity2.head_token
            else:
                head1 = edge.entity2.head_token
                head2 = edge.entity1.head_token

            sentence = edge.part.sentences[edge.sentence_id]

            for token in sentence:
                if not token.is_entity_part(edge.part) and any(x in token.word.lower() for x in self.loc_tokens):
                    location_word = True
                    if head1.features['id'] < token.features['id'] < head2.features['id']:
                        feature_name = self.mk_feature_name(self.prefix1, 'LocalizeWordInBetween')
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            if (location_word):
                feature_name = self.mk_feature_name(self.prefix2, 'locationWordFound')
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

            else:
                feature_name = self.mk_feature_name(self.prefix3, 'locationWordNotFound')
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
