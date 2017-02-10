from nalaf.learning.taggers import RelationExtractor, StubRelationExtractor
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
            selected_features_file=None,
            feature_generators=None,
            pipeline=None,
            execute_pipeline=True,
            model=None,
            **model_params):

        super().__init__(entity1_class, entity2_class, rel_type)

        self.sentence_distance = sentence_distance
        edge_generator = SentenceDistanceEdgeGenerator(entity1_class, entity2_class, rel_type, distance=self.sentence_distance)

        self.selected_features_file = selected_features_file

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
            # TODO set with this
            model_params["random_state"] = 2727
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

    def __init__(
        self,
        protein_id,
        localization_id,
        organism_id,
        uniprot_norm_id,
        string_norm_id,
        go_norm_id,
        taxonomy_norm_id,
        send_whole_once=True,
        host='http://127.0.0.1:5000'
    ):

        super().__init__([uniprot_norm_id, string_norm_id])

        self.send_whole_once = send_whole_once
        self.protein_id = protein_id
        self.localization_id = localization_id
        self.organism_id = organism_id
        self.uniprot_norm_id = uniprot_norm_id
        self.string_norm_id = string_norm_id
        self.go_norm_id = go_norm_id
        self.taxonomy_norm_id = taxonomy_norm_id
        self.host = host


    def annotate(self, dataset):
        """
        Primary method which will be called to set predicated annotations based on JSON response from STRING tagger.
        """

        for docid, document in dataset.documents.items():

            if self.send_whole_once:
                json_response = self.get_string_tagger_json_response(document.get_text())
                self.set_predicted_annotations(json_response, document, self.send_whole_once)
            else:
                for partid, part in document.parts.items():
                    json_response = self.get_string_tagger_json_response(part.text)
                    self.set_predicted_annotations(json_response, part, self.send_whole_once)

        # Verify entity offsets - No warnings should be displayed
        dataset.validate_entity_offsets()


    # gets String Tagger JSON response, by making a REST call.
    def get_string_tagger_json_response(self, payload):
        entry_point = self.host + "/annotate/post"
        response_status = None

        try:
            # Explicitly put all organisms in to force try to normalize all their proteins,
            # not only when their [organism] names appear together with a protein name
            entity_types = "-22,-3,9606,10090,3702,4932,4896,511145,6239,7227,7955"
            json_response = requests.post(entry_point, json=dict(text=payload, ids=entity_types, autodetect=True))
            response_status = json_response.status_code
            assert response_status == 200
            response = json_response.json()
            return response

        except Exception as e:
            server_running = self.server_is_running()
            howto_install = "https://github.com/juanmirocks/STRING-tagger-server"
            msg = "Failed call to STRING-tagger-server ({}; {}). Server running: {}. Response status: {}". \
                format(entry_point, howto_install, server_running, response_status)
            raise(msg, e)


    def server_is_running(self, host=None):
        """Return true if server is running"""
        host = host if host else self.host
        return urllib.request.urlopen(url).getcode() == 200


    # sets the predicted annotations of the parts or the whole text based on JSON response entity values
    def set_predicted_annotations(self, json_response, part_or_document, is_whole):

        entities = json_response["entities"]

        for index, entity in enumerate(entities):
            next_entity = entities[index + 1] if (index + 1) != len(entities) else None

            start = entity["start"]
            end = entity["end"]
            normalizations = entity["normalizations"]

            norm_ids = ""
            norm_id = ""
            uniprot_id = ""
            uniprot_ids = ""
            length = 1

            for norm in normalizations:

                if type(norm["type"]) is str:
                    if len(str(norm["id"]).split('|')):
                        uniprot_id = str(norm["id"]).split('|')[0]
                    else:
                        uniprot_id = str(norm["id"])
                else:
                    norm_id = str(norm["id"])

                if (next_entity and start == next_entity["start"] and end == next_entity["end"]):
                    if uniprot_id != "":
                        uniprot_ids += uniprot_id + ","
                    norm_ids += norm_id + ","

                else:
                    if uniprot_id != "":
                        uniprot_ids += uniprot_id
                    elif uniprot_ids.endswith(","):
                        uniprot_ids = uniprot_ids[:len(uniprot_ids)]

                    norm_ids += norm_id

            if is_whole is True:
                self.text_full(norm, norm_ids, uniprot_ids, part_or_document, start, end, length)
            else:
                self.text_part(norm, norm_ids, uniprot_ids, part_or_document, start, end)


    # helps to set the predicted annotations of the whole text based on JSON response entity values
    def text_full(self, norm, norm_ids, uniprot_ids, document, start, end, length):
        for partid, part in document.parts.items():

            offset = start - length
            text = part.text[offset:end - length + 1]

            if text != "":
                if str(norm["type"]) == "-3":
                    norm_dic = {self.taxonomy_norm_id: norm_ids}
                    normed_entity = Entity(class_id=self.organism_id, offset=offset, text=text, norm=norm_dic)
                elif str(norm["type"]) == "-22":
                    norm_dic = {self.go_norm_id: norm_ids}
                    normed_entity = Entity(class_id=self.localization_id, offset=offset, text=text, norm=norm_dic)
                else:
                    norm_dic = OrderedDict([(self.uniprot_norm_id, uniprot_ids), (self.string_norm_id, norm_ids)])
                    normed_entity = Entity(class_id=self.protein_id, offset=offset, text=text, norm=norm_dic)

                part.predicted_annotations.append(normed_entity)

                break

            length += len(part.text) + 1


    # helps to set the predicted annotations of the parts based on JSON response entity values
    def text_part(self, norm, norm_ids, uniprot_ids, part, start, end):

        offset = start - 1
        text = part.text[offset:end]

        if str(norm["type"]) == "-3":
            norm_dic = {self.taxonomy_norm_id: norm_ids}
            normed_entity = Entity(class_id=self.organism_id, offset=offset, text=text, norm=norm_dic)
        elif str(norm["type"]) == "-22":
            norm_dic = {self.go_norm_id: norm_ids}
            normed_entity = Entity(class_id=self.localization_id, offset=offset, text=text, norm=norm_dic)
        else:
            norm_dic = OrderedDict([(self.uniprot_norm_id, uniprot_ids), (self.string_norm_id, norm_ids)])
            normed_entity = Entity(class_id=self.protein_id, offset=offset, text=text, norm=norm_dic)

        part.predicted_annotations.append(normed_entity)


# usage of LoctextAnnotator:
# StringTagger creates entities (ner) and RelationExtraction gets those entities and creates relations (re)
class LocTextAnnotator(Tagger, RelationExtractor):

    def __init__(self, predict_classes, ner_kw_args, re_kw_args):
        # TODO better define init
        Tagger.__init__(self, predicts_classes=predict_classes)
        RelationExtractor.__init__(self, **re_kw_args)

        self.ner = StringTagger(**ner_kw_args)
        # TODO
        self.re = StubRelationExtractorFull(**re_kw_args)

    def annotate(self, dataset):
        self.ner.annotate(dataset)
        self.re.annotate(dataset)
        return dataset
