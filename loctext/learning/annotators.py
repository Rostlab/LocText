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
from loctext.learning.evaluations import are_go_parent_and_child, get_localization_name
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, repo_path


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
            use_predicted_entities=False,
            execute_pipeline=True,
            model=None,
            **model_params):

        super().__init__(entity1_class, entity2_class, rel_type)

        self.sentence_distance = sentence_distance
        edge_generator = SentenceDistanceEdgeGenerator(
            entity1_class, entity2_class, rel_type,
            distance=self.sentence_distance,
            use_gold=not use_predicted_entities,
            use_pred=use_predicted_entities,
        )

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
            model_params["tol"] = 1e-5

        if not model_params.get("random_state"):
            # TODO set with this
            model_params["random_state"] = 2727
            pass

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
        go_norm_id,
        taxonomy_norm_id,
        # Default: explicitly put all organisms in to force try to normalize all their proteins,
        # not only when their [organism] names appear together with a protein name
        # "-22,-3,9606,3702,4932,10090,4896,511145,6239,7227,7955",
        # Rather, put those organisms we know we collected documents from
        tagger_entity_types="-22,-3,9606,3702,4932",
        send_whole_once=True,
        filter_go_localizations=None,
        host='http://127.0.0.1:5000'
    ):

        super().__init__([protein_id, localization_id, organism_id])

        self.protein_id = protein_id
        self.localization_id = localization_id
        self.organism_id = organism_id
        self.uniprot_norm_id = uniprot_norm_id
        self.go_norm_id = go_norm_id
        self.taxonomy_norm_id = taxonomy_norm_id
        self.tagger_entity_types = tagger_entity_types
        self.send_whole_once = send_whole_once

        if filter_go_localizations is None:
            self.filter_go_localizations = {
                'GO:0009579',  # 1: thylakoid
                'GO:0005737',  # 2: cytoplasm
                'GO:0005811',  # 3: lipid particle
                'GO:0016020',  # 4: membrane
                'GO:0045202',  # 5: synapse
                'GO:0005856',  # 6: cytoskeleton
                'GO:0005694',  # 7: chromosome
                'GO:0005933',  # 8: cellular bud
                'GO:0005576',  # 9: extracellular region
                'GO:0005634',  # 10: nucleus
                'GO:0071944',  # 11: cell periphery
                'GO:0009986',  # 12: cell surface
                'GO:0042995',  # 13: cell projection
            }
        else:
            self.filter_go_localizations = set(filter_go_localizations)

        self.host = host


    def annotate(self, dataset):
        """
        Primary method which will be called to set predicated annotations based on JSON response from STRING tagger.
        """
        for docid, document in dataset.documents.items():
            if self.send_whole_once:
                document_text = document.get_text(separation=" ")
                tagger_annotations = self.get_string_tagger_json_response(document_text)

                index = 0
                part_length = 0
                extra_offset = 0
                num_of_tagger_entities = len(tagger_annotations)

                for partid, part in document.parts.items():
                    part_length += len(part.text) + 1  # 1 more due to the space separation
                    text = part.text

                    # For each part, insert the corresponding entities.
                    # Observation: Entities are in sequence, ex: If 1st entity belong to second part, then
                    #              2nd entity will not belong to first part and will belong to second or later parts.
                    while index < num_of_tagger_entities:
                        entity = tagger_annotations[index]
                        if entity['start'] < part_length:
                            index += 1

                            pred_entity = self.create_nalaf_entity(entity, text, offset_adjustment=(- extra_offset))
                            if pred_entity is not None:
                                part.predicted_annotations.append(pred_entity)

                        else:
                            extra_offset = part_length
                            break

            else:
                for partid, part in document.parts.items():
                    text = part.text
                    tagger_annotations = self.get_string_tagger_json_response(text)

                    for entity in tagger_annotations:
                        pred_entity = self.create_nalaf_entity(entity, text)
                        if pred_entity is not None:
                            part.predicted_annotations.append(pred_entity)

        dataset.validate_entity_offsets()


    def get_string_tagger_json_response(self, text):
        entry_point = self.host + "/annotate"
        response_status = None

        try:
            entity_types = self.tagger_entity_types
            json_response = requests.post(entry_point, json=dict(text=text, ids=entity_types, autodetect=True, output="simple"))
            response_status = json_response.status_code
            assert response_status == 200
            return json_response.json()

        except Exception as e:
            server_running = self.is_server_running()
            howto_install = "https://github.com/juanmirocks/STRING-tagger-server"
            msg = "Failed call to STRING-tagger-server ({}; {}). Server running: {}. Response status: {}". \
                format(entry_point, howto_install, server_running, response_status)

            raise Exception(msg, e)


    def is_server_running(self, host=None):
        """Return true if server is running"""
        host = host if host else self.host
        return urllib.request.urlopen(host).getcode() == 200


    def create_nalaf_entity(self, tagger_entity, original_text, offset_adjustment=0):

        offset = tagger_entity["start"] + offset_adjustment
        end = tagger_entity["end"] + offset_adjustment
        entity_text = original_text[offset:end]

        e_class_id = n_class_id = None
        norms = []
        organisms_proteins = {}

        for norm in tagger_entity["ids"]:
            # assumption: the e_class_id and n_class_id once set will not change

            norm_id = norm["id"]

            if norm["type"] == "-3":
                e_class_id = self.organism_id
                n_class_id = self.taxonomy_norm_id
                norms.append(norm_id)

            elif norm["type"] == "-22":
                try:
                    if any(are_go_parent_and_child(accepted_parent, norm_id) for accepted_parent in self.filter_go_localizations):
                        e_class_id = self.localization_id
                        n_class_id = self.go_norm_id
                        norms.append(norm_id)
                    else:
                        print("REJECT", norm_id, get_localization_name(norm_id))
                        pass  # reject

                except KeyError as e:
                    print("REJECT", norm_id, get_localization_name(norm_id))
                    pass  # reject

            elif norm["type"].startswith("uniprot_ac:"):
                organism = int(norm["type"].split(":")[1])
                prots = organisms_proteins.get(organism, set())
                prots.update({norm_id})
                organisms_proteins[organism] = prots

                e_class_id = self.protein_id
                n_class_id = self.uniprot_norm_id
                norms.append(norm_id)

            elif norm["type"].startswith("string_id:"):
                # Set e_class_id not to reject; this happens in the few cases the string id cannot be normalized to uniprot
                e_class_id = self.protein_id

        if not e_class_id:
            return None  # reject

        else:
            norms = set(norms)  # convert to set first just in case the original tagger returns repeated ids (happened)

            # Remove ambiguous ids; heuristic: different normalizations for a same organism are considered ambiguous
            for organism, proteins in organisms_proteins.items():
                if len(proteins) > 1:
                    for ambiguous_protein in proteins:
                        norms.remove(ambiguous_protein)

            if not norms:
                norms = None
            else:
                norms = ",".join(norms)

            if n_class_id:
                norms_dic = {n_class_id: norms}
            else:
                norms_dic = None

            pred_entity = Entity(class_id=e_class_id, offset=offset, text=entity_text, norm=norms_dic)

            return pred_entity


# usage of LoctextAnnotator:
# StringTagger creates entities (ner) and RelationExtraction gets those entities and creates relations (re)
class LocTextAnnotator(Tagger, RelationExtractor):

    def __init__(self, predict_classes, ner_kw_args, re_kw_args):
        # TODO better define init
        Tagger.__init__(self, predicts_classes=predict_classes)
        RelationExtractor.__init__(self, **re_kw_args)

        self.ner = StringTagger(**ner_kw_args)
        # TODO
        self.re = None

    def annotate(self, dataset):
        self.ner.annotate(dataset)
        self.re.annotate(dataset)
        return dataset
