from nalaf.learning.taggers import RelationExtractor
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.svmlight import SVMLightTreeKernels
from nalaf.features.relations import TokenFeatureGenerator
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.features.relations.sentence import NamedEntityCountFeatureGenerator
from nalaf.features.relations.context import LinearDistanceFeatureGenerator
from nalaf.features.relations.context import EntityOrderFeatureGenerator
from nalaf.features.relations.path import PathFeatureGenerator
from loctext.features.specific import LocationWordFeatureGenerator
from loctext.features.specific import ProteinWordFeatureGenerator



class LocTextBaselineRelationExtractor(RelationExtractor):

    def __init__(
        self,
        entity1_class,
        entity2_class,
        rel_type):

        super().__init__(entity1_class, entity2_class, rel_type)
        self.__annotator = StubSameSentenceRelationExtractor(entity1_class, entity2_class, rel_type)

    def annotate(self, corpus):
        return self.__annotator.annotate(corpus)


class LocTextRelationExtractor(RelationExtractor):

    @staticmethod
    def default_feature_generators(prot_e_id, loc_e_id, graphs=None):

        GRAPHS_CLOSURE_VARIABLE = {} if graphs is None else graphs

        return [
            LocationWordFeatureGenerator(
                loc_e_id,
                prefix1=2
            ),

            ProteinWordFeatureGenerator(
                GRAPHS_CLOSURE_VARIABLE,
                prefix_PWPE_bow=7,
                prefix_PWPE_bow_masked=9,
                prefix_PWPE_dep=10,
                prefix_protein_word_found=13,
                prefix_protein_not_word_found=14
            ),

            # TODO IntermediateTokensFeatureGenerator(feature_set, training_mode=train),

            LinearDistanceFeatureGenerator(
                distance=5,
                prefix_entity_linear_distance_greater_than=43,
                prefix_entity_linear_distance_lesser_than=44,
                # prefix_entity_linear_distance=45
            ),

            EntityOrderFeatureGenerator(
                # TODO change prefix names
            ),

            PathFeatureGenerator(
                GRAPHS_CLOSURE_VARIABLE
                # TODO change variables
                # TODO the used vars where not checked
            ),

            TokenFeatureGenerator(
                # TODO change variables
                # TODO the used vars where not checked
            )

            # TODO NamedEntityCountFeatureGenerator(
            #     prot_e_id,
            #     prefix=107),
            # TODO NamedEntityCountFeatureGenerator(
            #     loc_e_id,
            #     prefix=108),

            #
            # LocText original features as ordered by Madhukhar SP:
            #
            # EntityHeadTokenUpperCaseFeatureGenerator(feature_set, training_mode=train),
            # EntityHeadTokenDigitsFeatureGenerator(feature_set, training_mode=train),
            # EntityHeadTokenPunctuationFeatureGenerator(feature_set, training_mode=train),
            # BagOfWordsFeatureGenerator(feature_set, training_mode=train),
            # StemmedBagOfWordsFeatureGenerator(feature_set, training_mode=train),
        ]


    def __init__(
            self,
            entity1_class,
            entity2_class,
            rel_type,
            bin_model,
            pipeline=None,
            svmlight=None,
            execute_pipeline=True):

        super().__init__(entity1_class, entity2_class, rel_type)
        self.bin_model = bin_model
        self.svmlight = svmlight if svmlight else SVMLightTreeKernels(model_path=self.bin_model, use_tree_kernel=False)
        feature_generators = LocTextRelationExtractor.default_feature_generators(self.entity1_class, self.entity2_class)
        self.pipeline = pipeline if pipeline else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, feature_generators=feature_generators)
        self.execute_pipeline = execute_pipeline


    def annotate(self, corpus):
        if self.execute_pipeline:
            self.pipeline.execute(corpus, train=False)

        instancesfile = self.svmlight.create_input_file(corpus, 'predict', self.pipeline.feature_set)
        predictionsfile = self.svmlight.tag(instancesfile)
        self.svmlight.read_predictions(corpus, predictionsfile, threshold=0)

        return corpus
