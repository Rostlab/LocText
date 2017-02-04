from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.learning.taggers import Tagger, RelationExtractor
from nalaf.preprocessing.tokenizers import TmVarTokenizer, NLTK_TOKENIZER
from loctext.features.specific import IsProteinMarkerFeatureGenerator, LocalizationRelationsRatio, LocationWordFeatureGenerator, ProteinWordFeatureGenerator
from nalaf.features.relations.new.sentence import SentenceFeatureGenerator
from nalaf.features.relations.new.dependency import DependencyFeatureGenerator
from nalaf.features.relations.entityhead import EntityHeadTokenFeatureGenerator, EntityHeadTokenUpperCaseFeatureGenerator, EntityHeadTokenDigitsFeatureGenerator, EntityHeadTokenPunctuationFeatureGenerator, EntityHeadTokenChainFeatureGenerator
from nalaf.preprocessing.edges import SentenceDistanceEdgeGenerator
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.features.relations.sentence import NamedEntityCountFeatureGenerator
from nalaf.features.relations.entityhead import EntityHeadTokenUpperCaseFeatureGenerator, \
    EntityHeadTokenDigitsFeatureGenerator, EntityHeadTokenPunctuationFeatureGenerator
from nalaf.learning.taggers import Tagger
from loctext.util import UNIPROT_NORM_ID, STRING_NORM_ID
from nalaf.structures.data import Entity
import requests
import urllib.request


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
                #
                f_IW_bow_N_gram=None,  # 14
                f_IW_pos_N_gram=None,  # 15
                f_IW_tokens_count=None,  # 16
                f_IW_tokens_count_without_punct=None,  # 17
                #
                f_LD_bow_N_gram=None,  # 18
                f_LD_pos_N_gram=None,  # 19
                f_LD_tokens_count=None,  # 20
                f_LD_tokens_count_without_punct=21,  # 21
                #
                #
                f_PD_bow_N_gram=None,  # 22
                f_PD_pos_N_gram=None,  # 23
                f_PD_tokens_count=None,  # 24
                f_PD_tokens_count_without_punct=25,  # 25
                #
                f_PD_undirected_edges_N_gram=None,  # 26
                f_PD_directed_edges_N_gram=None,  # 27
                f_PD_full_N_gram=None,  # 28
                #
                #
            ),

            IsProteinMarkerFeatureGenerator(
                f_is_protein_marker=40,
            ),

            LocalizationRelationsRatio(
                f_localization_relation_ratio=42,
            ),

        ]

    # @staticmethod
    # def default_feature_generators(prot_e_id, loc_e_id, graphs=None):
    #
    #     GRAPHS_CLOSURE_VARIABLE = {} if graphs is None else graphs
    #
    #     return [
    #         LocationWordFeatureGenerator(
    #             loc_e_id,
    #             prefix1=2
    #         ),
    #
    #         ProteinWordFeatureGenerator(
    #             GRAPHS_CLOSURE_VARIABLE,
    #             prefix_PWPE_bow=7,
    #             prefix_PWPE_bow_masked=9,
    #             prefix_PWPE_dep=10,
    #             prefix_protein_word_found=13,
    #             prefix_protein_not_word_found=14
    #         ),
    #
    #         IntermediateTokensFeatureGenerator(
    #             prefix_fwd_pos_intermediate=34,
    #
    #             prefix_bkd_bow_intermediate=35,
    #             prefix_bkd_bow_intermediate_masked=36,
    #             prefix_bkd_stem_intermediate=37,
    #             prefix_bkd_pos_intermediate=38,
    #
    #             prefix_bow_intermediate=39,
    #             prefix_bow_intermediate_masked=40,
    #             prefix_stem_intermediate=41,
    #             prefix_pos_intermediate=42,
    #         ),
    #
    #         LinearDistanceFeatureGenerator(
    #             distance=5,
    #             prefix_entity_linear_distance_greater_than=43,
    #             prefix_entity_linear_distance_lesser_than=44,
    #             # prefix_entity_linear_distance=45
    #         ),
    #
    #         EntityOrderFeatureGenerator(
    #             prefix_order_entity1_entity2=46,
    #             prefix_order_entity2_entity1=47,
    #         ),
    #
    #         PathFeatureGenerator(
    #             GRAPHS_CLOSURE_VARIABLE,
    #
    #             token_feature_generator=TokenFeatureGenerator(
    #                 # prefix_txt=78,  # 73 in relna
    #                 prefix_pos=79,  # 74
    #                 prefix_masked_txt=77,  # 75
    #                 prefix_stem_masked_txt=81,  # 76
    #                 prefix_ann_type=80,  # 77
    #             ),
    #
    #             prefix_45_len_tokens=73,
    #             prefix_46_len=None,  # None
    #             prefix_47_word_in_path=None,  # None
    #             prefix_48_dep_forward=65,
    #             prefix_49_dep_reverse=66,
    #             prefix_50_internal_pos=67,
    #             prefix_51_internal_masked_txt=68,
    #             prefix_52_internal_txt=69,
    #             prefix_53_internal_stem=70,
    #             prefix_54_internal_dep_forward=71,
    #             prefix_55_internal_dep_reverse=72,
    #             prefix_56_token_path=64,
    #             prefix_57_dep_style_gram=60,
    #             prefix_58_edge_gram=None,  # None
    #             prefix_59_ann_edge_gram=None,  # None
    #             prefix_60_edge_directions=63,
    #             prefix_61_dep_1=49,
    #             prefix_62_masked_txt_dep_0=50,
    #             prefix_63_pos_dep_0=51,
    #             prefix_64_ann_type_1=52,
    #             prefix_65_dep_to_1=None,
    #             prefix_66_masked_txt_dep_to_0=53,
    #             prefix_67_pos_to=54,
    #             prefix_68_ann_type_2=55,
    #             prefix_69_gov_g_text=56,
    #             prefix_70_gov_g_pos=57,
    #             prefix_71_gov_anns=58,
    #             prefix_72_triple=59,
    #         ),
    #
    #         EntityHeadTokenUpperCaseFeatureGenerator(
    #             prefix_entity1_upper_case_middle=87.1,
    #             prefix_entity2_upper_case_middle=87.2,
    #         ),
    #
    #         EntityHeadTokenDigitsFeatureGenerator(
    #             prefix_entity1_has_hyphenated_digits=89.1,
    #             prefix_entity2_has_hyphenated_digits=89.2,
    #         ),
    #
    #         EntityHeadTokenPunctuationFeatureGenerator(
    #             prefix_entity1_has_hyphen=90.1,
    #             prefix_entity1_has_fslash=91.1,
    #             prefix_entity2_has_hyphen=90.2,
    #             prefix_entity2_has_fslash=91.2,
    #         ),
    #
    #         BagOfWordsFeatureGenerator(
    #             prefix_bow_text=2,
    #             prefix_ne_bow_count=3,
    #         ),
    #
    #         StemmedBagOfWordsFeatureGenerator(
    #             prefix_bow_stem=4
    #         ),
    #
    #         NamedEntityCountFeatureGenerator(
    #             prot_e_id,
    #             prefix=107
    #         ),
    #
    #         NamedEntityCountFeatureGenerator(
    #             loc_e_id,
    #             prefix=108
    #         )
    #     ]


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


class StringTagger(Tagger):
    def __init__(self, send_whole_once, protein_id, localization_id, organism_id,
                 uniprot_norm_id, go_norm_id, taxonomy_norm_id):
        self.send_whole_once = send_whole_once
        self.protein_id = protein_id
        self.localization_id = localization_id
        self.organism_id = organism_id
        self.uniprot_norm_id = uniprot_norm_id
        self.go_norm_id = go_norm_id
        self.taxonomy_norm_id = taxonomy_norm_id
        super().__init__([UNIPROT_NORM_ID, STRING_NORM_ID])

    # gets String Tagger JSON response, by making a REST call.
    def get_string_tagger_json_response(self, payload):
        base_url = "http://127.0.0.1:5000/annotate/post"
        try:
            json_response = requests.post(base_url, json=dict(text=payload, ids="-22,-3,9606"))
            json_response.status_code = 200
            response_data = json_response.json()
        except requests.exceptions.ConnectionError as err:
            print(
                "Sever is not running. For this application you need to install Docker https://docs.docker.com/engine/installation/ \n"
                "You only need to build the docker image once, like this: '$docker build -t tagger .' \n"
                "To run the docker image, you type this command: '$docker run -p 5000:5000 tagger'")
        return response_data

    # return true if server is running (for testing purposes)
    def server_is_running(host, url):
        return urllib.request.urlopen(url).getcode() == 200

    # sets the predicted annotations of the parts based on JSON response entity values.
    def set_predicted_annotations(self, json_response, part):

        entities = json_response["entities"]

        for index in range(len(entities)):
            start = entities[index]["start"]
            end = entities[index]["end"]
            normalizations = entities[index]["normalizations"]
            uniprot_id = ""
            entity_uniprot_ids = ""
            type_id = ""
            entity_type_ids = ""

            for norm in normalizations:

                if str(norm["type"]).isdigit():
                    type_id = str(norm["id"])
                else:
                    if len(str(norm["id"]).split('|')):
                        uniprot_id = str(norm["id"]).split('|')[0]
                    else:
                        uniprot_id = str(norm["id"])

            if(len(entities) != (index+1) and start == entities[index+1]["start"] and end == entities[index+1]["end"]):
                if uniprot_id != "":
                    entity_uniprot_ids += uniprot_id + ","
                entity_type_ids += type_id + ","

            else:
                if uniprot_id != "":
                    entity_uniprot_ids += uniprot_id
                elif entity_uniprot_ids.endswith(","):
                    entity_uniprot_ids = entity_uniprot_ids[:len(entity_uniprot_ids)]

                entity_type_ids += type_id

                if str(norm["type"]) == "-3":
                    norm_dictionary = {self.taxonomy_norm_id: entity_type_ids}
                    entity_dictionary = Entity(class_id=self.organism_id, offset=start-1, text=part.text[start-1:end],
                                               norm=norm_dictionary)
                elif str(norm["type"]) == "-22":
                    norm_dictionary = {self.go_norm_id: entity_type_ids}
                    entity_dictionary = Entity(class_id=self.localization_id, offset=start-1, text=part.text[start-1:end],
                                               norm=norm_dictionary)
                else:
                    norm_dictionary = {self.uniprot_norm_id: entity_uniprot_ids, STRING_NORM_ID: entity_type_ids}
                    entity_dictionary = Entity(class_id=self.protein_id, offset=start-1, text=part.text[start-1:end],
                                               norm=norm_dictionary)

                part.predicted_annotations.append(entity_dictionary)

                entity_uniprot_ids = ""
                entity_type_ids = ""

    # sets the predicted annotations of the whole text based on JSON response entity values
    def set_entities_of_whole_doc(self, json_response, document):

        entities = json_response["entities"]

        for index in range(len(entities)):
            start = entities[index]["start"]
            end = entities[index]["end"]
            normalizations = entities[index]["normalizations"]
            uniprot_id = ""
            entity_uniprot_ids = ""
            type_id = ""
            entity_type_ids = ""
            length = 1

            for norm in normalizations:

                if str(norm["type"]).isdigit():
                    type_id = str(norm["id"])
                else:
                    if len(str(norm["id"]).split('|')):
                        uniprot_id = str(norm["id"]).split('|')[0]
                    else:
                        uniprot_id = str(norm["id"])

            if(len(entities) != (index+1) and start == entities[index+1]["start"] and end == entities[index+1]["end"]):
                if uniprot_id != "":
                    entity_uniprot_ids += uniprot_id + ","
                entity_type_ids += type_id + ","

            else:
                if uniprot_id != "":
                    entity_uniprot_ids += uniprot_id
                elif entity_uniprot_ids.endswith(","):
                    entity_uniprot_ids = entity_uniprot_ids[:len(entity_uniprot_ids)]

                entity_type_ids += type_id

                for partId, part in document.parts.items():
                    text = part.text[start - length:end - length + 1]

                    if text != "":
                        if str(norm["type"]) == "-3":
                            norm_dictionary = {self.taxonomy_norm_id: entity_type_ids}
                            entity_dictionary = Entity(class_id=self.organism_id, offset=start-length, text=text,
                                                       norm=norm_dictionary)
                        elif str(norm["type"]) == "-22":
                            norm_dictionary = {self.go_norm_id: entity_type_ids}
                            entity_dictionary = Entity(class_id=self.localization_id, offset=start-length, text=text,
                                                       norm=norm_dictionary)
                        else:
                            norm_dictionary = {self.uniprot_norm_id: entity_uniprot_ids, STRING_NORM_ID: entity_type_ids}
                            entity_dictionary = Entity(class_id=self.protein_id, offset=start-length, text=text,
                                                       norm=norm_dictionary)

                        part.predicted_annotations.append(entity_dictionary)

                        break
                    length += len(part.text) + 1

                entity_uniprot_ids = ""
                entity_type_ids = ""


    # primary method which will be called to set predicated annotations based on JSON response from STRING tagger.
    def annotate(self, dataset):

        for docId, document in dataset.documents.items():

            if self.send_whole_once:
                # Note: dataset contains only the text content without separating into parts.
                json_response = self.get_string_tagger_json_response(document.get_text())
                self.set_entities_of_whole_doc(json_response, document)
            else:
                for partId, part in document.parts.items():
                    # Retrieve JSON response
                    json_response = self.get_string_tagger_json_response(part.text)

                    # Set entity information to part.predicated_annotations list
                    self.set_predicted_annotations(json_response, part)

        # Verify entity offsets - No warnings should be displayed
        dataset.validate_entity_offsets()
