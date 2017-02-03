from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.utils.graph import get_path, build_walks
from nalaf import print_debug
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, GO_NORM_ID, UNIPROT_NORM_ID, repo_path
from nalaf.features.stemming import ENGLISH_STEMMER
import pickle


class IsSpecificProteinType(EdgeFeatureGenerator):

    def __init__(
        self,
        c_protein_class=PRO_ID,
        c_set_protein_markers=None,
        #
        f_is_marker=None,
        f_is_enzyme=None,
        f_is_receptor=None,
        f_is_transporter=None,
    ):

        self.c_protein_class = c_protein_class

        if c_set_protein_markers is not None:
            self.c_set_protein_markers = self.c_set_protein_markers
        else:
            self.c_set_protein_markers = \
                {"GFP", "CYH2", "ALG2", "MSB2", "KSS1", "KRE11", "SER2"}

        self.f_is_marker = f_is_marker
        self.f_is_enzyme = f_is_enzyme
        self.f_is_receptor = f_is_receptor
        self.f_is_transporter = f_is_transporter

    def generate(self, corpus, f_set, is_train):
        for entity in corpus.entities():
            if entity.class_id == self.c_protein_class:
                entity.features["is_marker"] = entity.text in self.c_set_protein_markers
                entity.features["is_enzyme"] = any(t.word.endswith("ase") for t in entity.tokens)
                entity.features["is_receptor"] = any("recept" in t.word.lower() for t in entity.tokens)
                entity.features["is_transporter"] = any("transport" in t.word.lower() for t in entity.tokens)

                # Simple heuristic to know if some entities are abbreviations of another one
                # The protein x is abbreviation of protein y if they are written as: y (x)"
                # In the end, more generically, we call it a "synonym" relationship

                prev2 = entity.prev_tokens(entity.sentence, 2)
                next1 = entity.next_tokens(entity.sentence, 1)
                in_parenthesis = len(prev2) == 2 and prev2[-1].word == "(" and len(next1) == 1 and next1[0].word == ")"

                if (in_parenthesis):
                    prev_entity = prev2[0].get_entity(entity.part)

                    if prev_entity is not None and prev_entity.class_id == self.c_protein_class:
                        merged_binary_features = {key: (b1 or b2) for ((key, b1), (_, b2)) in zip(prev_entity.features.items(), entity.features.items())}
                        # TODO investigate
                        # prev_entity.features = merged_binary_features
                        # entity.features = merged_binary_features

                        prev_entity.features['synonym'] = entity
                        entity.features['synonym'] = prev_entity

        for edge in corpus.edges():
            sentence = edge.get_combined_sentence()

            protein = edge.entity1 if edge.entity1.class_id == self.c_protein_class else edge.entity2

            if protein.features["is_marker"]:
                self.add(f_set, is_train, edge, 'f_is_marker')

            if protein.features["is_enzyme"]:
                self.add(f_set, is_train, edge, 'f_is_enzyme')

            if protein.features["is_receptor"]:
                self.add(f_set, is_train, edge, 'f_is_receptor')

            if protein.features["is_transporter"]:
                self.add(f_set, is_train, edge, 'f_is_transporter')



class LocalizationRelationsRatios(EdgeFeatureGenerator):

    def __init__(
        self,
        c_localization_enty_class=LOC_ID,
        c_localization_norm_class=GO_NORM_ID,
        c_protein_enty_class=PRO_ID,
        c_protein_norm_class=UNIPROT_NORM_ID,
        #
        # ...constants...
        #
        f_corpus_unnormalized_total_background_loc_rels_ratios=None,
        f_corpus_normalized_total_background_loc_rels_ratios=None,
        #
        f_SwissProt_normalized_total_absolute_loc_rels_ratios=None,
        #
        #
        f_SwissProt_normalized_exists_relation=None,
    ):

        self.c_localization_enty_class = c_localization_enty_class
        self.c_localization_norm_class = c_localization_norm_class
        self.c_protein_enty_class = c_protein_enty_class
        self.c_protein_norm_class = c_protein_norm_class

        #

        path = repo_path(["resources", "features", "corpus_unnormalized_total_background_loc_rels_ratios.pickle"])
        with open(path, "rb") as f:
            self.c_corpus_unnormalized_total_background_loc_rels_ratios = pickle.load(f)

        path = repo_path(["resources", "features", "corpus_normalized_total_background_loc_rels_ratios.pickle"])
        with open(path, "rb") as f:
            self.c_corpus_normalized_total_background_loc_rels_ratios = pickle.load(f)

        path = repo_path(["resources", "features", "SwissProt_normalized_total_absolute_loc_rels_ratios.pickle"])
        with open(path, "rb") as f:
            self.c_SwissProt_normalized_total_absolute_loc_rels_ratios = pickle.load(f)

        path = repo_path(["resources", "features", "SwissProt_normalized_unique_absolute_loc_rels_ratios.pickle"])
        with open(path, "rb") as f:
            self.c_SwissProt_normalized_unique_absolute_loc_rels = pickle.load(f)

        #

        self.f_corpus_unnormalized_total_background_loc_rels_ratios = f_corpus_unnormalized_total_background_loc_rels_ratios
        self.f_corpus_normalized_total_background_loc_rels_ratios = f_corpus_normalized_total_background_loc_rels_ratios
        #
        self.f_SwissProt_normalized_total_absolute_loc_rels_ratios = f_SwissProt_normalized_total_absolute_loc_rels_ratios

        #
        #

        self.f_SwissProt_normalized_exists_relation = f_SwissProt_normalized_exists_relation


    def generate(self, corpus, f_set, is_train):
        for edge in corpus.edges():
            sentence = edge.get_combined_sentence()

            protein, localization = edge.entity1, edge.entity2
            if protein.class_id == self.c_localization_enty_class:
                protein, localization = localization, protein

            def add_f_ratio(f_key, ratio):
                ratio += 0.1  # Avoid absolute 0 weights
                self.add_with_value(f_set, is_train, edge, f_key, ratio)

            keyed_text = ENGLISH_STEMMER.stem(localization.text)
            ratio = self.c_corpus_unnormalized_total_background_loc_rels_ratios.get(keyed_text, 0)
            add_f_ratio("f_corpus_unnormalized_total_background_loc_rels_ratios", ratio)

            keyed_norm = list(localization.normalisation_dict.items())[0][1]
            ratio = self.c_corpus_normalized_total_background_loc_rels_ratios.get(keyed_norm, 0)
            add_f_ratio("f_corpus_normalized_total_background_loc_rels_ratios", ratio)

            loc_norm_id = localization.normalisation_dict.get(self.c_localization_norm_class, None)
            ratio = self.c_SwissProt_normalized_total_absolute_loc_rels_ratios.get(loc_norm_id, 0)
            add_f_ratio("f_SwissProt_normalized_total_absolute_loc_rels_ratios", ratio)

            pro_norm_ids_str = protein.normalisation_dict.get(self.c_protein_norm_class, None)
            if not pro_norm_ids_str:  # catches None or empty strings
                pro_norm_ids = []
            else:
                pro_norm_ids = pro_norm_ids_str.split(",")

                for pro_norm_id in pro_norm_ids:
                    go_id_rels = self.c_SwissProt_normalized_unique_absolute_loc_rels.get(pro_norm_id, set())

                    if loc_norm_id in go_id_rels:
                        # print("SUPPP", pro_norm_id, loc_norm_id)
                        self.add(f_set, is_train, edge, "f_SwissProt_normalized_exists_relation")


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
            sentence = edge.same_part.sentences[edge.same_sentence_id]
            protein_word_found = False

            for token in sentence:
                if token.is_entity_part(edge.same_part) and token.word.lower().find(self.keyword) >= 0:
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

                        path = get_path(token, head1, edge.same_part, edge.same_sentence_id, self.graphs)
                        if path == []:
                            path = [token, head1]


                        # TODO this may need to be indexed to the left, see original LocText: 5_
                        for tok in path:
                            feature_name = self.gen_prefix_feat_name("prefix_PWPE_bow", tok.word)
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                            feature_name = self.gen_prefix_feat_name("prefix_PWPE_pos", tok.features['pos'])
                            self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)

                            feature_name = self.gen_prefix_feat_name("prefix_PWPE_bow_masked", tok.masked_text(edge.same_part))
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

            sentence = edge.same_part.sentences[edge.same_sentence_id]

            for token in sentence:
                if not token.is_entity_part(edge.same_part) and any(x in token.word.lower() for x in self.loc_tokens):
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
