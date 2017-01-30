from nalaf.features.relations import EdgeFeatureGenerator
from nalaf.utils.graph import get_path, build_walks
from nalaf import print_debug
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from nalaf.features.stemming import ENGLISH_STEMMER


class IsProteinMarkerFeatureGenerator(EdgeFeatureGenerator):

    def __init__(
        self,
        c_protein_class=PRO_ID,
        c_set_protein_markers=None,
        #
        f_is_protein_marker=None,
    ):

        self.c_protein_class = c_protein_class

        if c_set_protein_markers is not None:
            self.c_set_protein_markers = self.c_set_protein_markers
        else:
            self.c_set_protein_markers = \
                {"GFP", "RFP", "CYH2", "ALG2", "MSB2", "KSS1", "KRE11", "SER2"}
                # review all treatment or enzymes
                # phosphatidylinositol-specific phospholipase C (PI-PLC -- treatment

        self.f_is_protein_marker = f_is_protein_marker

    def generate(self, corpus, f_set, is_train):
        for edge in corpus.edges():
            sentence = edge.get_combined_sentence()

            protein = edge.entity1 if edge.entity1.class_id == self.c_protein_class else edge.entity2

            is_protein_marker = protein.text in self.c_set_protein_markers

            if is_protein_marker:
                self.add(f_set, is_train, edge, 'f_is_protein_marker')


class LocalizationRelationsRatio(EdgeFeatureGenerator):

    def __init__(
        self,
        c_localization_class=LOC_ID,
        c_localization_relations_ratios=None,
        #
        f_localization_relation_ratio=None,
    ):

        self.c_localization_class = c_localization_class

        if c_localization_relations_ratios is not None:
            self.c_localization_relations_ratios = self.c_localization_relations_ratios
        else:
            self.c_localization_relations_ratios = {
                "chromoplast": 0.0,
                "integral outer membran": 0.0,
                "envelop": 0.0,
                "integral membran": 0.0,
                "project": 0.0,
                "mvb": 0.0,
                "cilia": 0.0,
                "chloroplast envelop": 0.0,
                "microtubule-organizing centr": 0.0,
                "nuclear por": 0.0,
                "secretori": 0.0,
                "extracellular": 0.0,
                "nucleolus": 0.0,
                "telomer": 0.0,
                "centrosom": 0.0,
                "nuclear envelop": 0.0,
                "lumen": 0.0,
                "nucleosom": 0.0,
                "thylakoid membran": 0.0,
                "cytoskeleton": 0.0,
                "bud": 0.3333333333333333,
                "microtubul": 0.3333333333333333,
                "er": 0.35714285714285715,
                "cell wal": 0.3888888888888889,
                "nucleoli": 0.5,
                "spindl": 0.5,
                "plastid": 0.5,
                "outer membran": 0.5,
                "golgi membran": 0.5,
                "chloroplast stroma": 0.5,
                "transmembran": 0.5,
                "kinetochor": 0.5,
                "chromosom": 0.5675675675675675,
                "endoplasmic reticulum": 0.5714285714285714,
                "melanosom": 0.625,
                "chloroplast": 0.6666666666666666,
                "surfac": 0.6666666666666666,
                "cytoplasm": 0.7777777777777778,
                "peroxisom": 0.8,
                "tonoplast": 0.8461538461538461,
                "membran": 0.9090909090909091,
                "mitochondrial matrix": 1.0,
                "vacuolar surfac": 1.0,
                "endoplasmic reticulum membran": 1.0,
                "mitochondria": 1.0,
                "lipid raft": 1.0,
                "cis-golgi stack": 1.0,
                "nuclei": 1.0,
                "apical plasma membran": 1.0,
                "mitochondrial inner membran": 1.0,
                "prekinetochor": 1.0,
                "chromocent": 1.0,
                "mitochondrial outer membran": 1.0,
                "thylakoid": 1.0,
                "spindle pole bodi": 1.0,
                "thylakoid membrane in chloroplast": 1.0,
                "cell membran": 1.0,
                "mitochondrial membran": 1.0,
                "vacuol": 1.0,
                "nuclear": 1.0344827586206897,
                "mitochondri": 1.1818181818181819,
                "secret": 1.2,
                "vacuolar": 1.2,
                "etioplast": 1.25,
                "heterochromat": 1.25,
                "lysosom": 1.25,
                "nuclear matrix": 1.25,
                "cytosol": 1.3333333333333333,
                "cell surfac": 1.3333333333333333,
                "nucleolar": 1.3333333333333333,
                "plasma membran": 1.4285714285714286,
                "nucleus": 1.4444444444444444,
                "cvt": 1.5,
                "cell boundari": 1.5,
                "centromer": 1.5294117647058822,
                "lipid particl": 1.5833333333333333,
                "tgn": 2.0,
                "basolater": 2.0,
                "ccvs": 2.0,
                "cell peripheri": 2.0,
                "clathrin-coated vesicl": 2.0,
                "intranuclear": 2.0,
                "synaps": 2.0,
                "outer mitochondrial membran": 2.0,
                "golgi": 2.0,
                "cellular protrus": 2.0,
                "cajal bodi": 2.0,
                "trans-golgi network": 2.0,
                "vacuolar membran": 2.4285714285714284,
                "endosom": 2.5833333333333335,
                "heterochromatin": 3.0,
                "peroxisomal membran": 3.0,
                "golgi apparatus": 4.0,
                "cell-surfac": 4.0,
                "subnuclear": 5.0,
                "extracellular matrix": 5.0,
                "intermembran": 7.0,
                "golgi stack": 7.0,
            }

        self.f_localization_relation_ratio = f_localization_relation_ratio

    def generate(self, corpus, f_set, is_train):
        for edge in corpus.edges():
            sentence = edge.get_combined_sentence()

            localization = edge.entity1 if edge.entity1.class_id == self.c_localization_class else edge.entity2
            norm_text = ENGLISH_STEMMER.stem(localization.text)

            ratio = self.c_localization_relations_ratios.get(norm_text, 0)
            ratio += 0.1  # Just to remove 0 weights, make minimally viable

            self.add_with_value(f_set, is_train, edge, 'f_localization_relation_ratio', ratio)


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
