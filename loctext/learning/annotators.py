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
from nalaf.preprocessing.edges import SimpleEdgeGenerator, SentenceDistanceEdgeGenerator
from nalaf import print_verbose, print_debug
from nalaf.learning.taggers import Tagger
from loctext.util import UNIPROT_NORM_ID, STRING_NORM_ID
#from nalaf.utils import MUT_CLASS_ID, PRO_CLASS_ID, PRO_REL_MUT_CLASS_ID, ENTREZ_GENE_ID, UNIPROT_ID
import difflib
from nalaf.utils.ncbi_utils import GNormPlus
from nalaf.utils.uniprot_utils import Uniprot
from nalaf.structures.data import Entity


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

        edge_generator = SentenceDistanceEdgeGenerator(entity1_class, entity2_class, rel_type, distance=0)
        self.pipeline = pipeline if pipeline else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, edge_generator=edge_generator, feature_generators=feature_generators)

        assert feature_generators == self.pipeline.feature_generators or feature_generators == [], str((feature_generators, self.pipeline.feature_generators))

        self.execute_pipeline = execute_pipeline

        # TODO this would require setting the default model_path
        self.svmlight = svmlight if svmlight else SVMLightTreeKernels(**svmlight_params)


    def annotate(self, target_corpus):
        if self.execute_pipeline:
            self.pipeline.execute(target_corpus, train=False)

        instancesfile = self.svmlight.create_input_file(target_corpus, 'predict', self.pipeline.feature_set)
        predictionsfile = self.svmlight.classify(instancesfile)
        self.svmlight.read_predictions(target_corpus, predictionsfile)

        return target_corpus


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

        edge_generator = SentenceDistanceEdgeGenerator(entity1_class, entity2_class, rel_type, distance=1)
        self.pipeline = pipeline if pipeline else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, edge_generator=edge_generator, feature_generators=feature_generators)

        assert feature_generators == self.pipeline.feature_generators or feature_generators == [], str((feature_generators, self.pipeline.feature_generators))

        self.execute_pipeline = execute_pipeline

        # TODO this would require setting the default model_path
        self.svmlight = svmlight if svmlight else SVMLightTreeKernels(**svmlight_params)


    def annotate(self, target_corpus):
        if self.execute_pipeline:
            self.pipeline.execute(target_corpus, train=False)

        instancesfile = self.svmlight.create_input_file(target_corpus, 'predict', self.pipeline.feature_set)
        predictionsfile = self.svmlight.classify(instancesfile)
        self.svmlight.read_predictions(target_corpus, predictionsfile)

        return target_corpus


    def feature_generators(self):
        return __class__.default_feature_generators(self.entity1_class, self.entity2_class)


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
        self.submodels = [self.ss_model, self.ds_model]


    def annotate(self, target_corpus):

        for model in self.submodels:
            model.annotate(target_corpus)

        return target_corpus

class StringTagger(Tagger):

    def __init__(self):
        super().__init__([STRING_NORM_ID, UNIPROT_NORM_ID])

    def __find_offset_adjustments(self, s1, s2, start_offset):
        return [(start_offset+i1, j2-j1-i2+i1)
                   for type, i1, i2, j1, j2  in difflib.SequenceMatcher(None, s1, s2).get_opcodes()
                   if type in ['replace', 'insert']]

    def tag(self, dataset, annotated=False, uniprot=False, process_only_abstract=True):
        doc_id, doc = dataset.documents.items()[0]
        part1 = doc.parts.values()[0]
        part1.text

        """
        :type dataset: nalaf.structures.data.Dataset
        :param annotated: if True then saved into annotations otherwise into predicted_annotations
        """
        with GNormPlus() as gnorm:
            for doc_id, doc in dataset.documents.items():
                if process_only_abstract:
                    genes, gnorm_title, gnorm_abstract = gnorm.get_genes_for_pmid(doc_id, postproc=True)

                    if uniprot:
                        with Uniprot() as uprot:
                            list_of_ids = gnorm.uniquify_genes(genes)
                            genes_mapping = uprot.get_uniprotid_for_entrez_geneid(list_of_ids)
                    else:
                        genes_mapping = {}

                    # find the title and the abstract
                    parts = iter(doc.parts.values())
                    title = next(parts)
                    abstract = next(parts)
                    adjustment_offsets = []
                    if title.text != gnorm_title:
                        adjustment_offsets += self.__find_offset_adjustments(title.text, gnorm_title, 0)
                    if abstract.text != gnorm_abstract:
                        adjustment_offsets += self.__find_offset_adjustments(abstract.text, gnorm_abstract, len(gnorm_title))

                    for start, end, text, gene_id in genes:
                        if 0 <= start < end <= len(title.text):
                            part = title
                        else:
                            part = abstract
                            # we have to readjust the offset since GnormPlus provides
                            # offsets for title and abstract together
                            offset = len(title.text) + 1
                            start -= offset
                            end -= offset

                        for adjustment_offset, adjustment in adjustment_offsets:
                            if start > adjustment_offset:
                                start -= adjustment

                        # todo discussion which confidence value for gnormplus because there is no value supplied
                        ann = Entity(class_id=UNIPROT_NORM_ID, offset=start, text=text, norm=norm_dict)
                        try:
                            norm_dict = {
                                STRING_NORM_ID: gene_id,
                                UNIPROT_NORM_ID: genes_mapping[gene_id]
                            }
                        except KeyError:
                            norm_dict = {STRING_NORM_ID: gene_id}

                        norm_string = ''  # todo normalized_text (stemming ... ?)
                        ann.normalisation_dict = norm_dict
                        ann.normalized_text = norm_string
                        if annotated:
                            part.annotations.append(ann)
                        else:
                            part.predicted_annotations.append(ann)
                else:
                    # todo this is not used for now anywhere, might need to be re-worked or excluded
                    # genes = gnorm.get_genes_for_text(part.text)
                    pass



