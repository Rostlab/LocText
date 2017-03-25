import pickle
from loctext.learning.annotators import LocTextDXModelRelationExtractor, LocTextCombinedModelRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf import print_verbose, print_debug
from loctext.learning.evaluations import is_in_swissprot, is_child_of_swissprot_annotation, accept_relation_uniprot_go, are_go_parent_and_child, get_localization_name, is_protein_in_swissprot, is_in_loctree3
from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Entity
from loctext.util import *
from collections import OrderedDict
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, repo_path
from loctext.learning.annotators import StringTagger
from collections import Counter
from nalaf.utils.download import DownloadArticle
from nalaf.structures.data import Dataset, Document, Part, Entity, Relation
from loctext.util import simple_parse_GO
import time


def parse_arguments(argv=[]):
    import argparse

    parser = argparse.ArgumentParser(description='dooh')

    parser.add_argument('--model', required=True, choices=["D0", "D1", "D0,D1", "D1,D0"])
    parser.add_argument('--predict_entities', default=None, choices=["9606,3702,4932", "9606", "3702", "4932"])
    parser.add_argument('--feature_generators', default='LocText', choices=["LocText", "default"])
    parser.add_argument('--save_model', default=None, help="Dir. path to save the trained model to")
    parser.add_argument('--load_model', default=None, help="File path to load a trained model from")

    parser.add_argument('--training_corpus', default="LocText", choices=["LocText", "LocText_FullTextsOnly"])
    parser.add_argument('--eval_corpus', required=False)
    parser.add_argument('--force_external_corpus_evaluation', default=False, action="store_true")
    parser.add_argument('--corpus_percentage', type=float, default=1.0, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')

    parser.add_argument('--evaluation_level', required=False, type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument('--evaluate_only_on_edges_plausible_relations', default=False, action='store_true')
    parser.add_argument('--cv_with_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)

    # # TODO clean and review how to set parameters for all different sentece models
    # parser.add_argument('--minority_class_ss_model', type=int, default=+1, choices=[-1, +1])
    # parser.add_argument('--majority_class_undersampling_ss_model', type=float, default=0.9, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    # parser.add_argument('--svm_hyperparameter_c_ss_model', action="store", default=0.0080)
    # parser.add_argument('--svm_threshold_ss_model', type=float, default=0.0)
    #
    # parser.add_argument('--minority_class_ds_model', type=int, default=+1, choices=[-1, +1])
    # parser.add_argument('--majority_class_undersampling_ds_model', type=float, default=0.07, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    # parser.add_argument('--svm_hyperparameter_c_ds_model', action="store", default=None)
    # parser.add_argument('--svm_threshold_ds_model', type=float, default=0.0)

    # ----------------------------------------------------------------------------------------------------

    def arg_bool(arg_value):
        FALSE = ['false', 'f', '0', 'n', 'no', 'none']
        return False if arg_value.lower() in FALSE else True

    def set_None_or_typed_argument(argument, expected_type):
        if not argument or argument == 'None':
            return None
        else:
            try:
                return expected_type(argument)
            except Exception as e:
                raise Exception("The argument {} must be of type {}".format(argument, str(expected_type)))

    def _select_submodel_params(annotator, args):

        if isinstance(annotator, LocTextDXModelRelationExtractor):
            return (args.minority_class_ss_model, args.majority_class_undersampling_ss_model, args.svm_hyperparameter_c_ss_model)

        raise AssertionError()

    # ----------------------------------------------------------------------------------------------------

    args = parser.parse_args(argv)

    args.evaluator = get_evaluator(args.evaluation_level, args.evaluate_only_on_edges_plausible_relations)
    args.predict_entities = [] if not args.predict_entities else [int(x) for x in args.predict_entities.split(",")]

    # args.svm_hyperparameter_c_ss_model = set_None_or_typed_argument(args.svm_hyperparameter_c_ss_model, float)
    # args.svm_hyperparameter_c_ds_model = set_None_or_typed_argument(args.svm_hyperparameter_c_ds_model, float)

    return args


def evaluate_with_argv(argv=[]):
    args = parse_arguments(argv)

    training_corpus = None
    eval_corpus = None

    if args.training_corpus and not args.load_model:
        training_corpus, eval_corpus = read_corpus(args.training_corpus, args.corpus_percentage, args.predict_entities, return_eval_corpus=True)

    if args.eval_corpus:
        eval_corpus = read_corpus(args.eval_corpus, args.corpus_percentage, args.predict_entities, return_eval_corpus=False)

    print_run_args(args, training_corpus, eval_corpus)
    result = evaluate(args, training_corpus, eval_corpus)
    print_run_args(args, training_corpus, eval_corpus)

    return result


def evaluate(args, training_corpus, eval_corpus):

    start = time.time()

    submodels = _select_annotator_submodels(args)
    are_features_already_extracted = False

    for submodel_name, submodel in submodels.items():
        print_debug("Training:", submodel_name)

        if not args.load_model:
            submodel.pipeline.execute(training_corpus, only_features=are_features_already_extracted)
            submodel.model.write_vector_instances(training_corpus, submodel.pipeline.feature_set)
            are_features_already_extracted = True

        annotator_gen_fun = (lambda training_set: train(args, submodel_name, training_set, submodel, execute_pipeline=False))

        if not (eval_corpus or args.save_model):
            # Do cross validation
            evaluations = Evaluations.cross_validate(annotator_gen_fun, training_corpus, args.evaluator, args.k_num_folds, use_validation_set=not args.cv_with_test_set)
            rel_evaluation = evaluations  # evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

        else:
            trained_annotator = annotator_gen_fun(training_corpus)

            if eval_corpus:
                submodel.pipeline.execute(eval_corpus, only_features=False)
                submodel.model.write_vector_instances(eval_corpus, submodel.pipeline.feature_set)

                trained_annotator(eval_corpus)

                if not args.force_external_corpus_evaluation and next(eval_corpus.relations(), None):
                    # The corpus has annotated relationships: run a normal performance evaluation
                    rel_evaluation = args.evaluator.evaluate(eval_corpus)
                else:
                    # Else, write in a file the extracted relationships
                    rel_evaluation = write_external_evaluation_results(args, eval_corpus)

            else:
                rel_evaluation = "Model saved into folder: " + args.save_model

    end = time.time()
    print("Time for total evaluation:", (end - start))

    return rel_evaluation


def train(args, submodel_name, training_set, submodel, execute_pipeline):
    if execute_pipeline:
        submodel.pipeline.execute(training_set)

    if not args.load_model:
        # Train if not loaded model
        submodel.model.train(training_set)

    if args.save_model:
        timestamp = time.time()
        model_filename = "{}_{}_{}.bin".format(submodel_name, ",".join([str(x) for x in predict_entities]), timestamp)
        model_path = os.path.join(args.save_model, model_filename)

        with open(model_path, "wb") as f:
            pickle.dump(submodel.model, f)
            print("Model saved to: ", model_path)

    return submodel.annotate


def _select_annotator_submodels(args):
    # WARN: we should read the class ids from the corpus
    pro_id = PRO_ID
    loc_id = LOC_ID
    rel_id = REL_PRO_LOC_ID

    indirect_feature_generators = {
        "LocText": None,  # Uses annotator's default
        "default": []  # Uses RelationExtractionPipeline's default

    }.get(args.feature_generators)

    submodels = OrderedDict()  # Order may matter
    submodels_names = args.model.split(",")

    binary_model = None

    if args.load_model:
        if len(submodels_names) > 1:
            raise NotImplementedError("No current support for loading multple trained models")
        else:
            with open(args.load_model, "rb") as f:
                binary_model = pickle.load(f)

    execute_pipeline = False  # not binary_model -- decide later

    for name in submodels_names:
        # TODO get here: minority_class, majority_class_undersampling, svm_hyperparameter_c = ...

        if "D0" == name:
            if args.predict_entities:
                selected_features_file = repo_path("resources", "features", "selected", "0_True_LinearSVC_C=2.0-1487943476.673364-NAMES.py")

                submodels[name] = LocTextDXModelRelationExtractor(
                    pro_id, loc_id, rel_id,
                    sentence_distance=0,
                    selected_features_file=selected_features_file,
                    feature_generators=indirect_feature_generators,
                    use_predicted_entities=True,
                    execute_pipeline=execute_pipeline,
                    model=binary_model,
                    #
                    preprocess=True,
                    #
                    class_weight=None,
                    kernel='linear',
                    C=1,
                )

            else:
                selected_features_file = repo_path("resources", "features", "selected", "0_False_LinearSVC-1486292275.065055-NAMES.py")

                submodels[name] = LocTextDXModelRelationExtractor(
                    pro_id, loc_id, rel_id,
                    sentence_distance=0,
                    selected_features_file=selected_features_file,
                    feature_generators=indirect_feature_generators,
                    use_predicted_entities=False,
                    execute_pipeline=execute_pipeline,
                    model=binary_model,
                    #
                    preprocess=True,
                    #
                    class_weight=None,
                    kernel='linear',
                    C=1,
                )

        if "D1" == name:
            selected_features_file = repo_path("resources", "features", "selected", "1_False_LinearSVC-1486481526.730234-NAMES.py")

            submodels[name] = LocTextDXModelRelationExtractor(
                pro_id, loc_id, rel_id,
                sentence_distance=1,
                selected_features_file=selected_features_file,
                feature_generators=indirect_feature_generators,
                use_predicted_entities=len(args.predict_entities) > 0,
                execute_pipeline=execute_pipeline,
                model=binary_model,
                #
                preprocess=True,
                #
                class_weight=None,
                kernel='linear',
                C=1,
            )

    assert submodels, "No model given!"

    return submodels


def write_external_evaluation_results(args, eval_corpus):
    if len(args.predict_entities) == 0:
        args.predict_entities = [9606]
    else:
        assert len(args.predict_entities) == 1

    organism_id = args.predict_entities[0]

    if organism_id == 4932:
        # All STRING Tagger yeast normalizations are to the strain 559292
        organism_id = 559292

    macro_counter = Counter()
    micro_counter = {}

    for docid, doc in eval_corpus.documents.items():

        for rel in doc.predicted_relations():

            if rel.entity1.class_id == PRO_ID and rel.entity2.class_id == LOC_ID:
                e1, e2 = rel.entity1, rel.entity2
            elif rel.entity1.class_id == LOC_ID and rel.entity2.class_id == PRO_ID:
                e1, e2 = rel.entity2, rel.entity1
            else:
                raise AssertionError(("Cannot be", rel))

            e1s = e1.normalisation_dict.get(UNIPROT_NORM_ID, None)
            if e1s is not None:
                e1s = e1s = filter(None, e1s.split(","))
            else:
                e1s = []

            e2s = e2.normalisation_dict.get(GO_NORM_ID, None)
            if e2s is not None:
                e2s = e2s = filter(None, e2s.split(","))
            else:
                e2s = []

            pairs = zip(e1s, e2s)

            for e1, e2 in pairs:
                rel_key = (e1, e2)

                rel_key_docid_counters = micro_counter.get(rel_key, Counter())

                if docid not in rel_key_docid_counters:  # this rel_key was not in this counter before
                    macro_counter.update({rel_key})

                rel_key_docid_counters.update({docid})
                micro_counter[rel_key] = rel_key_docid_counters

    with open(args.eval_corpus + "_" + "relations.tsv", "w") as f:

        header = ["UniProtAC", "LOC_GO", "LOC_NAME", "In SwissProt", "ChildOf SwissProt", "In LocTree3", "Confirmed", "Num Docs"]
        max_num_docs = len(micro_counter[macro_counter.most_common(1)[0][0]])
        header = header + (["PMID"] * max_num_docs)
        f.write("\t".join(header) + "\n")

        for rel_key, count in macro_counter.most_common():
            u_ac, go = rel_key

            if is_protein_in_swissprot(u_ac, organism_id):
                loc_name = get_localization_name(go)

                inSwissProt = str(is_in_swissprot(u_ac, go, organism_id))
                childSwissProt = str(is_child_of_swissprot_annotation(u_ac, go, organism_id))
                inLocTree3 = str(is_in_loctree3(u_ac, go, organism_id))
                confirmed = ""

                cols = [u_ac, go, loc_name, inSwissProt, childSwissProt, inLocTree3, confirmed, str(count)]
                cols = cols + [docid for docid, _ in micro_counter[rel_key].most_common()]
                f.write("\t".join(cols) + "\n")

    rel_evaluation = macro_counter.most_common(100)

    return rel_evaluation


def read_corpus(corpus_name, corpus_percentage=1.0, predict_entities=None, return_eval_corpus=False):
    import os
    from nalaf.utils.readers import HTMLReader
    from nalaf.utils.annotation_readers import AnnJsonAnnotationReader

    start = time.time()

    if isinstance(predict_entities, str):
        predict_entities = list(filter(None, predict_entities.split(",")))

    __corpora_dir = repo_path("resources", "corpora")

    if corpus_name in ["LocText", "LocText_v2"]:  # With reviewed normalizations (8 new protein normalizations made by Tanya; no other modifications)
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_annjson_with_normalizations_latest_5_feb_2017/')

    elif corpus_name == "LocText_FullTextsOnly":
        dir_html = os.path.join(__corpora_dir, 'LocText/FullTextsOnly/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/FullTextsOnly/')

    elif corpus_name == "LocText_v1":  # With normalizations; normalizations done in excel sheet
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_annjson_with_normalizations/')

    elif corpus_name == "LocText_v0":  # Original as annotated from tagtog, without normalizations at all
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_master_json/pool/')

    elif corpus_name in ["NewDiscoveries_9606", "NewDiscoveries_3702", "NewDiscoveries_4932"]:

        if corpus_name == "NewDiscoveries_9606":
            pmids_file_path = os.path.join(repo_path("resources", "evaluation", "human_pubmed_result.txt"))

        elif corpus_name == "NewDiscoveries_3702":
            pmids_file_path = os.path.join(repo_path("resources", "evaluation", "arabidopsis_pubmed_result.txt"))

        elif corpus_name == "NewDiscoveries_4932":
            pmids_file_path = os.path.join(repo_path("resources", "evaluation", "yeast_pubmed_result.txt"))

        dir_html = None
        corpus = Dataset()

        with DownloadArticle() as PMID_DL:
            with open(pmids_file_path) as f:
                for pmid in f:
                    pmid = pmid.strip()

                    for _, doc in PMID_DL.download([pmid]):
                        try:
                            doc.get_text()  # Had problems with few documents and needs investigation; empty docs?
                            corpus.documents[pmid] = doc
                        except Exception:
                            pass

    else:
        raise AssertionError(("Corpus not recognized: ", corpus_name))

    if dir_html is not None:
        corpus = HTMLReader(dir_html).read()

        if corpus_name == "LocText" or corpus_name.startswith("LocText_v"):
            # Remove PMCs, full-text
            del corpus.documents["PMC3596250"]
            del corpus.documents["PMC2192646"]
            del corpus.documents["PMC2483532"]
            del corpus.documents["PMC2847216"]

        if dir_annjson is not None:

            AnnJsonAnnotationReader(
                dir_annjson,
                read_only_class_id=[PRO_ID, LOC_ID, ORG_ID],
                read_relations=True,
                delete_incomplete_docs=False).annotate(corpus)

    eval_corpus = None

    if (corpus_percentage < 1.0):
        corpus, eval_corpus = corpus.percentage_split(corpus_percentage)

    if predict_entities:
        tagger_entity_types = "-22,-3," + ",".join([str(x) for x in predict_entities])

        STRING_TAGGER = StringTagger(PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, tagger_entity_types=tagger_entity_types, send_whole_once=True)

        STRING_TAGGER.annotate(corpus)
        if return_eval_corpus and eval_corpus:
            STRING_TAGGER.annotate(eval_corpus)

    end = time.time()
    print("Time for reading the corpus", corpus_name, (end - start))

    if return_eval_corpus:
        return (corpus, eval_corpus)

    else:
        return corpus


def get_evaluator(evaluation_level, evaluate_only_on_edges_plausible_relations=False, normalization_penalization="no"):

    if evaluation_level == 1:
        ENTITY_MAP_FUN = Entity.__repr__
        RELATION_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 2:
        ENTITY_MAP_FUN = 'lowercased'
        RELATION_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 3:
        ENTITY_MAP_FUN = DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_fun'](
            # WARN: we should read the class ids from the corpus
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization,
        )
        RELATION_ACCEPT_FUN = str.__eq__

    elif evaluation_level == 4:
        ENTITY_MAP_FUN = DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_fun'](
            # WARN: we should read the class ids from the corpus
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations=normalization_penalization
        )
        RELATION_ACCEPT_FUN = accept_relation_uniprot_go

    else:
        raise AssertionError(evaluation_level)

    evaluator = DocumentLevelRelationEvaluator(
        rel_type=REL_PRO_LOC_ID,
        entity_map_fun=ENTITY_MAP_FUN,
        relation_accept_fun=RELATION_ACCEPT_FUN,
        evaluate_only_on_edges_plausible_relations=evaluate_only_on_edges_plausible_relations,
    )

    return evaluator


def print_run_args(args, training_corpus, eval_corpus):
    print()
    print("Run Arguments: ")
    for key, value in sorted((vars(args)).items()):
        print("\t{} = {}".format(key, value))

    print()
    print_corpus_hard_core_stats("Training", training_corpus)
    print_corpus_hard_core_stats("Evaluation", eval_corpus)


def print_corpus_hard_core_stats(name, corpus):
    if corpus:
        print(name + " corpus stats:")
        print("\t#documents: {}".format(len(corpus)))
        print("\t#relations total: {}".format(sum(1 for r in corpus.relations())))
        print("\t#relations prot<-->loc: {}".format(sum(1 for r in corpus.relations() if r.class_id == REL_PRO_LOC_ID)))
        entity_counter = Counter()
        for e in corpus.entities():
            entity_counter.update([e.class_id])
        print("\t#entities: {}".format(entity_counter))

        print_corpus_pipeline_dependent_stats(corpus)
        print()


def print_corpus_pipeline_dependent_stats(corpus):
    T = 0
    P = 0
    N = 0

    for e in corpus.edges():
        T += 1
        if e.is_relation():
            P += 1
        else:
            N += 1

    if T > 0:
        # Assumes the sentences and edges have been generated (through relations_pipeline)
        print("\t#sentences: {}".format(len(list(corpus.sentences()))))
        print("\t#instances (edges): {} -- #P={} vs. #N={}".format(T, P, N))
        print("\t#plausible relations from edges: {}".format(len(list(corpus.plausible_relations_from_generated_edges()))))
        print("\t#features: {}".format(next(corpus.edges()).features_vector.shape[1]))

    return (P, N)


if __name__ == "__main__":
    import sys
    ret = evaluate_with_argv(sys.argv[1:])
    print(ret)
