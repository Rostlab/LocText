from nalaf.features.relations import EdgeFeatureGenerator

class LocationWordFeatureGenerator(EdgeFeatureGenerator):
    """
    Check each sentence for the presence of location words if the sentence
    contains an edge. These location words include ['location', 'localize'].

    :param feature_set: the feature set for the dataset
    :type feature_set: nalaf.structures.data.FeatureDictionary
    :param training_mode: indicates whether the mode is training or testing
    :type training_mode: bool
    """
    def __init__(self, loc_e_id, prefix1, prefix2=None, prefix3):
        self.loc_e_id = loc_id
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.prefix3 = prefix3
        pass


    def generate(self, dataset, feature_set, is_training_mode):
        for edge in dataset.edges():
            location_word = False
            if edge.entity1.class_id != self.loc_e_id:  #  'e_1' (protein)
                head1 = edge.entity1.head_token
                head2 = edge.entity2.head_token
            else:
                head1 = edge.entity2.head_token
                head2 = edge.entity1.head_token
            sentence = edge.part.sentences[edge.sentence_id]
            for token in sentence:
                if not token.is_entity_part(edge.part) and ('location' in token.word.lower() or 'localize' in token.word.lower()):
                    location_word = True
                    if head1.features['id'] < token.features['id'] < head2.features['id']:
                        feature_name = '88_localize_word_in_between_[0]'
                        self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            if (location_word):
                feature_name = '89_location_word_found_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
            else:
                feature_name = '90_location_word_not_found_[0]'
                self.add_to_feature_set(feature_set, is_training_mode, edge, feature_name)
