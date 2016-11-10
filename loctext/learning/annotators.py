from nalaf.learning.taggers import RelationExtractor
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.svmlight import SVMLightTreeKernels
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from loctext.features.specific import LocationWordFeatureGenerator
from loctext.features.specific import ProteinWordFeatureGenerator
from nalaf.features.relations import TokenFeatureGenerator
from nalaf.features.relations.context import LinearDistanceFeatureGenerator
from nalaf.features.relations.context import EntityOrderFeatureGenerator
from nalaf.features.relations.context import IntermediateTokensFeatureGenerator
from nalaf.features.relations.path import PathFeatureGenerator
from nalaf.features.relations.sentence import NamedEntityCountFeatureGenerator, BagOfWordsFeatureGenerator, StemmedBagOfWordsFeatureGenerator
from nalaf.features.relations.entityhead import EntityHeadTokenUpperCaseFeatureGenerator, EntityHeadTokenDigitsFeatureGenerator, EntityHeadTokenPunctuationFeatureGenerator
from nalaf.preprocessing.edges import SimpleEdgeGenerator, SimpleD1EdgeGenerator


class LocTextSSmodelRelationExtractor(RelationExtractor):

    def __init__(
            self,
            entity1_class,
            entity2_class,
            rel_type,
            feature_generators=None,
            pipeline=None,
            execute_pipeline=True,
            svmlight=None,
            **svmlight_params):

        super().__init__(entity1_class, entity2_class, rel_type)

        if pipeline:
            feature_generators = pipeline.feature_generators
        elif feature_generators is not None:  # Trick: if [], this will use pipeline's default generators
            feature_generators = feature_generators
        else:
            feature_generators = self.feature_generators()


        edge_generator = SimpleEdgeGenerator(entity1_class, entity2_class, rel_type)
        self.pipeline = pipeline if pipeline else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, edge_generator=edge_generator, feature_generators=feature_generators)

        assert feature_generators == self.pipeline.feature_generators or feature_generators == [], str((feature_generators, self.pipeline.feature_generators))

        self.execute_pipeline = execute_pipeline

        # TODO this would require setting the default model_path
        self.svmlight = svmlight if svmlight else SVMLightTreeKernels(**svmlight_params)


    def annotate(self, corpus):
        if self.execute_pipeline:
            self.pipeline.execute(corpus, train=False)

        instancesfile = self.svmlight.create_input_file(corpus, 'predict', self.pipeline.feature_set)
        predictionsfile = self.svmlight.classify(instancesfile)
        self.svmlight.read_predictions(corpus, predictionsfile)

        return corpus


    def feature_generators(self):
        return __class__.default_feature_generators(self.entity1_class, self.entity2_class)


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

            IntermediateTokensFeatureGenerator(
                prefix_fwd_pos_intermediate=34,

                prefix_bkd_bow_intermediate=35,
                prefix_bkd_bow_intermediate_masked=36,
                prefix_bkd_stem_intermediate=37,
                prefix_bkd_pos_intermediate=38,

                prefix_bow_intermediate=39,
                prefix_bow_intermediate_masked=40,
                prefix_stem_intermediate=41,
                prefix_pos_intermediate=42,
            ),

            LinearDistanceFeatureGenerator(
                distance=5,
                prefix_entity_linear_distance_greater_than=43,
                prefix_entity_linear_distance_lesser_than=44,
                # prefix_entity_linear_distance=45
            ),

            EntityOrderFeatureGenerator(
                prefix_order_entity1_entity2=46,
                prefix_order_entity2_entity1=47,
            ),

            PathFeatureGenerator(
                GRAPHS_CLOSURE_VARIABLE,

                token_feature_generator=TokenFeatureGenerator(
                    # prefix_txt=78,  # 73 in relna
                    prefix_pos=79,  # 74
                    prefix_masked_txt=77,  # 75
                    prefix_stem_masked_txt=81,  # 76
                    prefix_ann_type=80,  # 77
                ),

                prefix_45_len_tokens=73,
                prefix_46_len=None,  # None
                prefix_47_word_in_path=None,  # None
                prefix_48_dep_forward=65,
                prefix_49_dep_reverse=66,
                prefix_50_internal_pos=67,
                prefix_51_internal_masked_txt=68,
                prefix_52_internal_txt=69,
                prefix_53_internal_stem=70,
                prefix_54_internal_dep_forward=71,
                prefix_55_internal_dep_reverse=72,
                prefix_56_token_path=64,
                prefix_57_dep_style_gram=60,
                prefix_58_edge_gram=None,  # None
                prefix_59_ann_edge_gram=None,  # None
                prefix_60_edge_directions=63,
                prefix_61_dep_1=49,
                prefix_62_masked_txt_dep_0=50,
                prefix_63_pos_dep_0=51,
                prefix_64_ann_type_1=52,
                prefix_65_dep_to_1=None,
                prefix_66_masked_txt_dep_to_0=53,
                prefix_67_pos_to=54,
                prefix_68_ann_type_2=55,
                prefix_69_gov_g_text=56,
                prefix_70_gov_g_pos=57,
                prefix_71_gov_anns=58,
                prefix_72_triple=59,
            ),

            EntityHeadTokenUpperCaseFeatureGenerator(
                prefix_entity1_upper_case_middle=87.1,
                prefix_entity2_upper_case_middle=87.2,
            ),

            EntityHeadTokenDigitsFeatureGenerator(
                prefix_entity1_has_hyphenated_digits=89.1,
                prefix_entity2_has_hyphenated_digits=89.2,
            ),

            EntityHeadTokenPunctuationFeatureGenerator(
                prefix_entity1_has_hyphen=90.1,
                prefix_entity1_has_fslash=91.1,
                prefix_entity2_has_hyphen=90.2,
                prefix_entity2_has_fslash=91.2,
            ),

            BagOfWordsFeatureGenerator(
                prefix_bow_text=2,
                prefix_ne_bow_count=3,
            ),

            StemmedBagOfWordsFeatureGenerator(
                prefix_bow_stem=4
            ),

            NamedEntityCountFeatureGenerator(
                prot_e_id,
                prefix=107
            ),

            NamedEntityCountFeatureGenerator(
                loc_e_id,
                prefix=108
            )
        ]


class LocTextDSmodelRelationExtractor(RelationExtractor):

    def __init__(
            self,
            entity1_class,
            entity2_class,
            rel_type,
            feature_generators=None,
            pipeline=None,
            execute_pipeline=True,
            svmlight=None,
            **svmlight_params):

        super().__init__(entity1_class, entity2_class, rel_type)

        if pipeline:
            feature_generators = pipeline.feature_generators
        elif feature_generators is not None:  # Trick: if [], this will use pipeline's default generators
            feature_generators = feature_generators
        else:
            feature_generators = self.feature_generators()

        edge_generator = SimpleD1EdgeGenerator(entity1_class, entity2_class, rel_type)
        self.pipeline = pipeline if pipeline else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, edge_generator=edge_generator, feature_generators=feature_generators)

        assert feature_generators == self.pipeline.feature_generators or feature_generators == [], str((feature_generators, self.pipeline.feature_generators))

        # #
        # # TODO force it for now to be empty
        #
        # TODO see below in feature_generators
        # #
        self.pipeline.feature_generators = []

        self.execute_pipeline = execute_pipeline

        # TODO this would require setting the default model_path
        self.svmlight = svmlight if svmlight else SVMLightTreeKernels(**svmlight_params)


    def annotate(self, corpus):
        #
        # TODO do something!
        #

        if self.execute_pipeline:
            self.pipeline.execute(corpus, train=False)

        print("\n\n\n**************PUTA PUTA PUTA *\n\n\n", "\n", "\n", len(list(corpus.edges())), "\n", [e.target for e in corpus.edges()].count(1), "\n")

        #
        # instancesfile = self.svmlight.create_input_file(corpus, 'predict', self.pipeline.feature_set)
        # predictionsfile = self.svmlight.classify(instancesfile)
        # self.svmlight.read_predictions(corpus, predictionsfile)

        return corpus


    def feature_generators(self):
        # TODO define DS features
        # return __class__.default_feature_generators(self.entity1_class, self.entity2_class)
        return []
        # return LocTextSSmodelRelationExtractor.default_feature_generators(self.entity1_class, self.entity2_class)


    @staticmethod
    def default_feature_generators(prot_e_id, loc_e_id, graphs=None):

        GRAPHS_CLOSURE_VARIABLE = {} if graphs is None else graphs

        return []


class LocTextCombinedModelRelationExtractor(RelationExtractor):

    def __init__(
            self,
            entity1_class,
            entity2_class,
            rel_type,
            ss_model,
            ds_model):

        super().__init__(entity1_class, entity2_class, rel_type)

        self.ss_model = ss_model
        self.ds_model = ds_model
        self.submodels = [self.ss_model] # , self.ds_model]


    def annotate(self, corpus):

        # TODO fix & clean

        for model in self.submodels:
            model.annotate(corpus)

        # self.ss_model(corpus)

        return corpus
