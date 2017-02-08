from nalaf.learning.taggers import RelationExtractor, StubRelationExtractor, StubRelationExtractorFull
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.learning.taggers import Tagger, RelationExtractor
from nalaf.preprocessing.tokenizers import TmVarTokenizer
from loctext.features.specific import IsSpecificProteinType, LocalizationRelationsRatios, LocationWordFeatureGenerator, ProteinWordFeatureGenerator
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
from collections import OrderedDict


class LocTextDXModelRelationExtractor(RelationExtractor):

    def __init__(
            self,
            entity1_class,
            entity2_class,
            rel_type,
            sentence_distance=0,
            feature_generators=None,
            pipeline=None,
            execute_pipeline=True,
            model=None,
            **model_params):

        super().__init__(entity1_class, entity2_class, rel_type)

        self.sentence_distance = sentence_distance
        edge_generator = SentenceDistanceEdgeGenerator(entity1_class, entity2_class, rel_type, distance=self.sentence_distance)

        if pipeline:
            feature_generators = pipeline.feature_generators
        elif feature_generators is not None:  # Trick: if [], this will use pipeline's default generators
            feature_generators = feature_generators
        else:
            feature_generators = self.feature_generators()

        self.pipeline = pipeline if pipeline \
            else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, tokenizer=TmVarTokenizer(), edge_generator=edge_generator, feature_generators=feature_generators)

        assert feature_generators == self.pipeline.feature_generators or feature_generators == [], str((feature_generators, self.pipeline.feature_generators))

        self.execute_pipeline = execute_pipeline

        # With the following two settings we try force the model to always give the same results between runs
        # and avoid slight variations due to different random generators initializations

        if not model_params.get("tol"):
            # As of 2017-Feb-7, default in SVC is 1e-3: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            model_params["tol"] = 1e-8

        if not model_params.get("random_state"):
            # TODO model_params["random_state"] = 2727
            pass

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
                f_LD_is_negated=None,  # 103
                #
                #
                f_PD_bow_N_gram=22,  # 22
                f_PD_pos_N_gram=23,  # 23
                f_PD_tokens_count=None,  # 24
                f_PD_tokens_count_without_punct=25,  # 25
                f_PD_is_negated=None,  # 104
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
            # Explicitly put all organisms in to force try to normalize all their proteins,
            # not only when their [organism] names appear together with a protein name
            entity_types = "-22,-3,9606,10090,3702,4932,4896,511145,6239,7227,7955"
            json_response = requests.post(base_url, json=dict(text=payload, ids=entity_types))
            json_response.status_code = 200
            response_data = json_response.json()
        except requests.exceptions.ConnectionError as err:
            print(
                "Sever is not running. For this application you need to install Docker "
                "https://docs.docker.com/engine/installation/ \n"
                "You only need to build the docker image once, like this: '$docker build -t tagger .' \n"
                "To run the docker image, you type this command: '$docker run -p 5000:5000 tagger'")
        return response_data

    # return true if server is running (for testing purposes)
    def server_is_running(host, url):
        return urllib.request.urlopen(url).getcode() == 200

    # helps to set the predicted annotations of the whole text based on JSON response entity values
    def text_full(self, norm, entity_type_ids, entity_uniprot_ids, document, start, end, length):
        for partId, part in document.parts.items():
            text = part.text[start - length:end - length + 1]

            if text != "":
                if str(norm["type"]) == "-3":
                    norm_dictionary = {self.taxonomy_norm_id: entity_type_ids}
                    entity_dictionary = Entity(class_id=self.organism_id, offset=start - length, text=text,
                                               norm=norm_dictionary)
                elif str(norm["type"]) == "-22":
                    norm_dictionary = {self.go_norm_id: entity_type_ids}
                    entity_dictionary = Entity(class_id=self.localization_id, offset=start - length, text=text,
                                               norm=norm_dictionary)
                else:
                    norm_dictionary = OrderedDict(
                        [(self.uniprot_norm_id, entity_uniprot_ids), (STRING_NORM_ID, entity_type_ids)])
                    entity_dictionary = Entity(class_id=self.protein_id, offset=start - length, text=text,
                                               norm=norm_dictionary)

                part.predicted_annotations.append(entity_dictionary)

                break
            length += len(part.text) + 1

    # helps to set the predicted annotations of the parts based on JSON response entity values
    def text_part(self, norm, entity_type_ids, entity_uniprot_ids, part, start, end):

        if str(norm["type"]) == "-3":
            norm_dictionary = {self.taxonomy_norm_id: entity_type_ids}
            entity_dictionary = Entity(class_id=self.organism_id, offset=start - 1, text=part.text[start - 1:end],
                                       norm=norm_dictionary)
        elif str(norm["type"]) == "-22":
            norm_dictionary = {self.go_norm_id: entity_type_ids}
            entity_dictionary = Entity(class_id=self.localization_id, offset=start - 1, text=part.text[start - 1:end],
                                       norm=norm_dictionary)
        else:
            norm_dictionary = OrderedDict(
                [(self.uniprot_norm_id, entity_uniprot_ids), (STRING_NORM_ID, entity_type_ids)])
            entity_dictionary = Entity(class_id=self.protein_id, offset=start - 1, text=part.text[start - 1:end],
                                       norm=norm_dictionary)

        part.predicted_annotations.append(entity_dictionary)

    # sets the predicted annotations of the parts or the whole text based on JSON response entity values
    def set_predicted_annotations(self,json_response, part_or_document, isWhole):

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

                if type(norm["type"]) is str:
                    if len(str(norm["id"]).split('|')):
                        uniprot_id = str(norm["id"]).split('|')[0]
                    else:
                        uniprot_id = str(norm["id"])
                else:
                    type_id = str(norm["id"])

                if (len(entities) != (index + 1) and start == entities[index + 1]["start"] and end ==
                    entities[index + 1]["end"]):
                    if uniprot_id != "":
                        entity_uniprot_ids += uniprot_id + ","
                    entity_type_ids += type_id + ","

                else:
                    if uniprot_id != "":
                        entity_uniprot_ids += uniprot_id
                    elif entity_uniprot_ids.endswith(","):
                        entity_uniprot_ids = entity_uniprot_ids[:len(entity_uniprot_ids)]

                    entity_type_ids += type_id

                    if isWhole == True:
                        self.text_full(norm, entity_type_ids, entity_uniprot_ids, part_or_document, start, end, length)
                    else:
                        self.text_part(norm, entity_type_ids, entity_uniprot_ids, part_or_document, start, end)

                entity_uniprot_ids = ""
                entity_type_ids = ""


    # primary method which will be called to set predicated annotations based on JSON response from STRING tagger.
    def annotate(self, dataset):

        for docId, document in dataset.documents.items():

            if self.send_whole_once:
                # Note: dataset contains only the text content without separating into parts.
                json_response = self.get_string_tagger_json_response(document.get_text())
                self.set_predicted_annotations(json_response, document, self.send_whole_once)
            else:
                for partId, part in document.parts.items():
                    # Retrieve JSON response
                    json_response = self.get_string_tagger_json_response(part.text)

                    # Set entity information to part.predicated_annotations list
                    self.set_predicted_annotations(json_response, part, self.send_whole_once)

        # Verify entity offsets - No warnings should be displayed
        dataset.validate_entity_offsets()


# usage of LoctextAnnotator:
# StringTagger creates entities (ner) and RelationExtraction gets those entities and creates relations (re)
class LocTextAnnotator(Tagger, RelationExtractor):

    def __init__(self, dataset, predict_classes, **re_kw_args):

        Tagger.__init__(self, predicts_classes=predict_classes)
        RelationExtractor.__init__(self, **re_kw_args)

        self.dataset = dataset

    # annotate for named entity recognition
    def ner_annotate(self, **ner_kw_args):
        StringTagger(**ner_kw_args).annotate(self.dataset)

    # annotate for relation extraction
    def re_annotate(self, **re_kw_args):
        return StubRelationExtractorFull(**re_kw_args).annotate
