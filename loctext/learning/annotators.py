from nalaf.learning.taggers import RelationExtractor
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.preprocessing.tokenizers import TmVarTokenizer, NLTK_TOKENIZER
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from loctext.features.specific import IsSpecificProteinType, LocalizationRelationsRatios, LocationWordFeatureGenerator, ProteinWordFeatureGenerator
from nalaf.features.relations import TokenFeatureGenerator
from nalaf.features.relations.context import LinearDistanceFeatureGenerator
from nalaf.features.relations.context import EntityOrderFeatureGenerator
from nalaf.features.relations.context import IntermediateTokensFeatureGenerator
from nalaf.features.relations.path import PathFeatureGenerator
from nalaf.features.relations.sentence import NamedEntityCountFeatureGenerator, BagOfWordsFeatureGenerator, StemmedBagOfWordsFeatureGenerator
from nalaf.features.relations.new.sentence import SentenceFeatureGenerator
from nalaf.features.relations.new.dependency import DependencyFeatureGenerator
from nalaf.features.relations.entityhead import EntityHeadTokenFeatureGenerator, EntityHeadTokenUpperCaseFeatureGenerator, EntityHeadTokenDigitsFeatureGenerator, EntityHeadTokenPunctuationFeatureGenerator, EntityHeadTokenChainFeatureGenerator
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator
from nalaf import print_verbose, print_debug


class LocTextSSmodelRelationExtractor(RelationExtractor):

    def __init__(
            self,
            entity1_class,
            entity2_class,
            rel_type,
            feature_generators=None,
            pipeline=None,
            execute_pipeline=True,
            model=None,
            **model_params):

        super().__init__(entity1_class, entity2_class, rel_type)

        if pipeline:
            feature_generators = pipeline.feature_generators
        elif feature_generators is not None:  # Trick: if [], this will use pipeline's default generators
            feature_generators = feature_generators
        else:
            feature_generators = self.feature_generators()

        edge_generator = SentenceDistanceEdgeGenerator(entity1_class, entity2_class, rel_type, distance=0)
        self.pipeline = pipeline if pipeline else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, tokenizer=TmVarTokenizer(), edge_generator=edge_generator, feature_generators=feature_generators)

        assert feature_generators == self.pipeline.feature_generators or feature_generators == [], str((feature_generators, self.pipeline.feature_generators))

        self.execute_pipeline = execute_pipeline

        # TODO this would require setting the default model_path
        self.model = model if model else SklSVM(**model_params)


    def annotate(self, target_corpus):
        if self.execute_pipeline:
            self.pipeline.execute(target_corpus, train=False)

        self.model.annotate(target_corpus)

        return target_corpus


    def feature_generators(self):
        return __class__.default_feature_generators(self.entity1_class, self.entity2_class)


    @staticmethod
    def default_feature_generators(prot_e_id, loc_e_id):

        return [
            SentenceFeatureGenerator(
                f_counts_individual=1.1,  # 1.1
                f_counts_total=1.2,  # 1.2
                f_counts_in_between_individual=None,  # 2.1
                f_counts_in_between_total=2.2,  # 2.2

                f_order=3,  # 3

                f_bow=None,  # 4
                f_pos=None,  # 5

                f_tokens_count=None,  # 6
                f_tokens_count_before=None,  # 7
                f_tokens_count_after=None,  # 8

                f_sentence_is_negated=None,  # 105
                f_main_verbs=None,  # 106

                f_entity1_count=None,  # 110
                f_entity2_count=None,  # 111
                f_diff_sents_together_count=None,  # 112
            ),

            DependencyFeatureGenerator(
                # Hyper parameters
                h_ow_size=3,  # outer window size
                h_ow_grams=[1, 2],
                h_iw_size=0,  # inner window size
                h_iw_grams=[],
                h_ld_grams=[1, 2, 3],
                h_pd_grams=[1, 2, 3],
                # Feature keys/names
                f_OW_bow_N_gram=None,  # 10
                f_OW_pos_N_gram=None,  # 11
                f_OW_tokens_count=None,  # 12
                f_OW_tokens_count_without_punct=None,  # 13
                f_OW_is_negated=None,  # 101
                #
                f_IW_bow_N_gram=None,  # 14
                f_IW_pos_N_gram=None,  # 15
                f_IW_tokens_count=None,  # 16
                f_IW_tokens_count_without_punct=None,  # 17
                f_IW_is_negated=None,  # 102
                #
                f_LD_bow_N_gram=18,  # 18
                f_LD_pos_N_gram=19,  # 19
                f_LD_tokens_count=None,  # 20
                f_LD_tokens_count_without_punct=21,  # 21
                f_LD_is_negated=103,  # 103
                #
                #
                f_PD_bow_N_gram=22,  # 22
                f_PD_pos_N_gram=23,  # 23
                f_PD_tokens_count=None,  # 24
                f_PD_tokens_count_without_punct=25,  # 25
                f_PD_is_negated=104,  # 104
                #
                f_PD_undirected_edges_N_gram=26,  # 26
                f_PD_directed_edges_N_gram=None,  # 27
                f_PD_full_N_gram=None,  # 28
                #
                #
            ),

            IsSpecificProteinType(
                f_is_marker=40,
                f_is_enzyme=41,
                f_is_receptor=None,
                f_is_transporter=None,
            ),

            LocalizationRelationsRatios(
                f_corpus_unnormalized_total_background_loc_rels_ratios=50,  # 50
                f_corpus_normalized_total_background_loc_rels_ratios=None,  # 51
                f_SwissProt_normalized_total_absolute_loc_rels_ratios=None,  # 52
                f_SwissProt_normalized_total_background_loc_rels_ratios=None,
                #
                f_SwissProt_normalized_exists_relation=58,
            ),

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
            model=None,
            **model_params):

        super().__init__(entity1_class, entity2_class, rel_type)

        if pipeline:
            feature_generators = pipeline.feature_generators
        elif feature_generators is not None:  # Trick: if [], this will use pipeline's default generators
            feature_generators = feature_generators
        else:
            feature_generators = self.feature_generators()

        edge_generator = SentenceDistanceEdgeGenerator(entity1_class, entity2_class, rel_type, distance=1)
        self.pipeline = pipeline if pipeline else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, edge_generator=edge_generator, feature_generators=feature_generators)

        assert feature_generators == self.pipeline.feature_generators or feature_generators == [], str((feature_generators, self.pipeline.feature_generators))

        self.execute_pipeline = execute_pipeline

        # TODO this would require setting the default model_path
        self.model = model if model else SklSVM(**model_params)


    def annotate(self, target_corpus):
        if self.execute_pipeline:
            self.pipeline.execute(target_corpus, train=False)

        self.model.annotate(target_corpus)

        return target_corpus


    def feature_generators(self):
        return __class__.default_feature_generators(self.entity1_class, self.entity2_class)


    @staticmethod
    def default_feature_generators(prot_e_id, loc_e_id, graphs=None):
        from loctext.features import DS

        GRAPHS_CLOSURE_VARIABLE = {} if graphs is None else graphs

        return [
            # Comment from Shrikant:
            # TODO ...Commenting follwing two function calls increases the overall Fscore by 0.3 points...

            # Trigram
            DS.AnyNGramFeatureGenerator(
                n_gram=3,
                #
                prefix_pos=3
            ),

            # Bigram
            DS.AnyNGramFeatureGenerator(
                n_gram=2,
                #
                prefix_bow=5,
                prefix_bow_masked=6,
                prefix_pos=7,
                prefix_stem=8
            ),

            DS.PatternFeatureGenerator(
                e1_class=prot_e_id,
                e2_class=loc_e_id,
                #
                prefix_ProtVerbWord=39,
                prefix_LocVerbWord=41,
                prefix_ProtVerbWordLocVerbWord=43,
                prefix_WordVerbProtLocVerbWord=45,
            ),

            DS.SameWordFeatureGenerator(
                prefix_sameWordsSamePOS=48,
                prefix_sameStemSamePOS=50,
            ),

            DS.LocEntityFeatureGenerator(
                localization_class_id=loc_e_id,
                prefix_localizationVerb=51
            ),

            DS.IndividualSentencesFeatureGenerator(
                prefix_sentence_1_POS=54,
                #
                prefix_sentence_2_POS=57
            ),

            DS.IntermediateTokenFeatureGenerator(
                prefix_fwdPOSIntermeditate=61,
                prefix_bkwdPOSIntermeditate=65,
                prefix_unorderedPOSIntermeditate=69,
            ),

            DS.LinearDistanceFeatureGenerator(
                # TODO such a small distance doesn't seem to make sense
                distance_threshold=5,
                #
                prefix_entityLinearDistGreaterThan=70,
                prefix_entityLinearDistLessThanOrEqual=71,
                prefix_entityLinearDist=72,
                prefix_entityLinearDistOffsets=72.1,
            ),

            #####

            EntityHeadTokenFeatureGenerator(

            ),

            EntityHeadTokenUpperCaseFeatureGenerator(
                prefix_entity1_upper_case_start=111.1,
                prefix_entity2_upper_case_start=111.2,
                prefix_entity1_upper_case_middle=112.1,
                prefix_entity2_upper_case_middle=112.1,
            ),

            EntityHeadTokenDigitsFeatureGenerator(
                prefix_entity1_has_digits=113.1,
                prefix_entity2_has_digits=113.2,
                prefix_entity1_has_hyphenated_digits=114.1,
                prefix_entity2_has_hyphenated_digits=114.2,
            ),

            EntityHeadTokenPunctuationFeatureGenerator(
                prefix_entity1_has_hyphen=115.1,
                prefix_entity2_has_hyphen=115.2,
                prefix_entity1_has_fslash=116.1,
                prefix_entity2_has_fslash=116.2,
            ),

            EntityHeadTokenChainFeatureGenerator(

            ),

            DS.BowFeatureGenerator(
                prefix_bow_of_tokens_and_entities=129,
            ),

            NamedEntityCountFeatureGenerator(
                prot_e_id,
                prefix=130
            ),

            NamedEntityCountFeatureGenerator(
                loc_e_id,
                prefix=131
            ),
        ]


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
        self.submodels = [self.ss_model, self.ds_model]


    def annotate(self, target_corpus):

        for model in self.submodels:
            model.annotate(target_corpus)

        return target_corpus
