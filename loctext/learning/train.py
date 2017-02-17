from loctext.learning.annotators import LocTextDXModelRelationExtractor, LocTextCombinedModelRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf import print_verbose, print_debug
from loctext.learning.evaluations import relation_accept_uniprot_go, _go_ids_accept_single
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

import pickle

def parse_arguments(argv=[]):
    import argparse

    parser = argparse.ArgumentParser(description='dooh')

    parser.add_argument('--model', required=True, choices=["D0", "D1", "D0,D1", "D1,D0"])

    parser.add_argument('--corpus', default="LocText", choices=["LocText"])
    parser.add_argument('--corpus_percentage', type=float, required=True, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')
    parser.add_argument('--test_corpus', required=False, choices=["SwissProt", "NewDiscoveries", "LocText"])
    parser.add_argument('--evaluation_level', type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument('--evaluate_only_on_edges_plausible_relations', default=False, action='store_true')
    parser.add_argument('--predict_entities', default=False, type=bool, choices=[True, False])

    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)

    parser.add_argument('--feature_generators', default='LocText', choices=["LocText", "default"])
    parser.add_argument('--use_tk', default=False, action='store_true')

    parser.add_argument('--minority_class_ss_model', type=int, default=+1, choices=[-1, +1])
    parser.add_argument('--majority_class_undersampling_ss_model', type=float, default=0.9, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    parser.add_argument('--svm_hyperparameter_c_ss_model', action="store", default=0.0080)
    parser.add_argument('--svm_threshold_ss_model', type=float, default=0.0)

    # TODO clean and review how to set parameters for all different sentece models
    parser.add_argument('--minority_class_ds_model', type=int, default=+1, choices=[-1, +1])
    parser.add_argument('--majority_class_undersampling_ds_model', type=float, default=0.07, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    parser.add_argument('--svm_hyperparameter_c_ds_model', action="store", default=None)
    parser.add_argument('--svm_threshold_ds_model', type=float, default=0.0)

    args = parser.parse_args(argv)

    if args.evaluation_level == 1:
        ENTITY_MAP_FUN = Entity.__repr__
        RELATION_ACCEPT_FUN = str.__eq__
    elif args.evaluation_level == 2:
        ENTITY_MAP_FUN = 'lowercased'
        RELATION_ACCEPT_FUN = str.__eq__
    elif args.evaluation_level == 3:
        ENTITY_MAP_FUN = DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations="soft"
        )
        RELATION_ACCEPT_FUN = str.__eq__
    elif args.evaluation_level == 4:
        ENTITY_MAP_FUN = DocumentLevelRelationEvaluator.COMMON_ENTITY_MAP_FUNS['normalized_fun'](
            {
                PRO_ID: UNIPROT_NORM_ID,
                LOC_ID: GO_NORM_ID,
                ORG_ID: TAXONOMY_NORM_ID,
            },
            penalize_unknown_normalizations="soft"
        )
        RELATION_ACCEPT_FUN = relation_accept_uniprot_go

    args.evaluator = DocumentLevelRelationEvaluator(
        rel_type=REL_PRO_LOC_ID,
        entity_map_fun=ENTITY_MAP_FUN,
        relation_accept_fun=RELATION_ACCEPT_FUN,
        evaluate_only_on_edges_plausible_relations=args.evaluate_only_on_edges_plausible_relations,
    )

    def set_None_or_typed_argument(argument, expected_type):
        if not argument or argument == 'None':
            return None
        else:
            try:
                return expected_type(argument)
            except Exception as e:
                raise Exception("The argument {} must be of type {}".format(argument, str(expected_type)))

    args.svm_hyperparameter_c_ss_model = set_None_or_typed_argument(args.svm_hyperparameter_c_ss_model, float)
    args.svm_hyperparameter_c_ds_model = set_None_or_typed_argument(args.svm_hyperparameter_c_ds_model, float)

    return args


def parse_arguments_string(arguments=""):
    return parse_arguments(arguments.split("\s+"))


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

    for name in submodels_names:
        # TODO get here: minority_class, majority_class_undersampling, svm_hyperparameter_c = _select_submodel_params(annotator, args)

        if "D0" == name:
            if args.predict_entities:
                selected_features_file = "/Users/juanmirocks/Work/hck/LocText/tmp/0_True_LinearSVC-1486839732.581865-NAMES.log"
            else:
                selected_features_file = "/Users/juanmirocks/Work/hck/LocText/tmp/0_False_LinearSVC-1486292275.065055-NAMES.log"
                # selected_features.remove("LocalizationRelationsRatios::50_corpus_unnormalized_total_background_loc_rels_ratios_[0]")

            submodels[name] = LocTextDXModelRelationExtractor(
                pro_id, loc_id, rel_id,
                sentence_distance=0,
                selected_features_file=selected_features_file,
                feature_generators=indirect_feature_generators,
                use_predicted_entities=args.predict_entities,
                execute_pipeline=False,
                model=None,
                classification_threshold=args.svm_threshold_ss_model,
                use_tree_kernel=args.use_tk,
                preprocess=True,
                #
                class_weight=None,
                kernel='linear',
                C=1,
            )

        if "D1" == name:
            selected_features_file = "/Users/juanmirocks/Work/hck/LocText/tmp/1_False_LinearSVC-1486481526.730234-NAMES.log"

            submodels[name] = LocTextDXModelRelationExtractor(
                pro_id, loc_id, rel_id,
                sentence_distance=1,
                selected_features_file=selected_features_file,
                feature_generators=indirect_feature_generators,
                use_predicted_entities=args.predict_entities,
                execute_pipeline=False,
                model=None,
                classification_threshold=args.svm_threshold_ss_model,
                use_tree_kernel=args.use_tk,
                preprocess=True,
                #
                class_weight=None,
                kernel='linear',
                C=1,
            )

    assert submodels, "No model given!"

    return submodels


def _select_submodel_params(annotator, args):

    if isinstance(annotator, LocTextDXModelRelationExtractor):
        return (args.minority_class_ss_model, args.majority_class_undersampling_ss_model, args.svm_hyperparameter_c_ss_model)

    raise AssertionError()


def train(training_set, args, submodel, execute_pipeline):
    if execute_pipeline:
        submodel.pipeline.execute(training_set, train=True)

    submodel.model.train(training_set)

    return submodel.annotate


def evaluate(training_corpus, test_corpus, args):
    evaluator = args.evaluator

    submodels = _select_annotator_submodels(args)
    execute_only_features = False

    for submodel_name, submodel in submodels.items():
        print("Executing :", submodel_name)

        submodel.pipeline.execute(training_corpus, train=True, only_features=execute_only_features)
        execute_only_features = True
        selected_features = unpickle_beautified_file(submodel.selected_features_file)
        submodel.model.set_allowed_feature_names(submodel.pipeline.feature_set, selected_features)
        submodel.model.write_vector_instances(training_corpus, submodel.pipeline.feature_set)

        annotator_gen_fun = (lambda training_set: train(training_set, args, submodel, execute_pipeline=False))

        if not args.test_corpus:
            evaluations = Evaluations.cross_validate(annotator_gen_fun, training_corpus, evaluator, args.k_num_folds, use_validation_set=not args.use_test_set)
            rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")
        else:
            submodel.pipeline.execute(test_corpus, train=False, only_features=False)
            selected_features = unpickle_beautified_file(submodel.selected_features_file)
            submodel.model.set_allowed_feature_names(submodel.pipeline.feature_set, selected_features)
            submodel.model.write_vector_instances(test_corpus, submodel.pipeline.feature_set)

            annotator = annotator_gen_fun(training_corpus)
            annotator(test_corpus)

            macro_counter = Counter()
            micro_counter = {}

            with open(repo_path(["resources", "features", "SwissProt_relations.pickle"]), "rb") as f:
                SWISSPROT_RELATIONS = pickle.load(f)

            GO_TREE = simple_parse_GO.simple_parse(repo_path(["resources", "ontologies", "go-basic.cellular_component.latest.obo"]))

            for docid, doc in test_corpus.documents.items():

                for rel in doc.predicted_relations():
                    e1s = filter(None, rel.entity1.normalisation_dict.get(UNIPROT_NORM_ID, "").split(","))
                    e2s = filter(None, rel.entity2.normalisation_dict.get(GO_NORM_ID, "").split(","))
                    if (rel.entity2.class_id <= rel.entity1.class_id):
                        pairs = zip(e2s, e1s)
                    else:
                        pairs = zip(e1s, e2s)

                    for e1, e2 in pairs:
                        rel_key = (e1, e2)

                        rel_key_docid_counters = micro_counter.get(rel_key, Counter())

                        if docid not in rel_key_docid_counters:  # this rel_key was not in this counter before
                            macro_counter.update({rel_key})

                        rel_key_docid_counters.update({docid})
                        micro_counter[rel_key] = rel_key_docid_counters

            with open(args.test_corpus + "_" + "relations.tsv", "w") as f:

                header = ["# Type", "UniProtAC", "LOC_GO", "LOC_NAME", "In SwissProt", "Child SwissProt", "Confirmed", "Num Docs"]
                max_num_docs = len(micro_counter[macro_counter.most_common(1)[0][0]])
                header = header + (["PMID"] * max_num_docs)
                f.write("\t".join(header) + "\n")

                for rel_key, count in macro_counter.most_common():
                    u_ac, go = rel_key
                    name = GO_TREE.get(go, ("", "", ""))[0]
                    inSwissProt = str(go in SWISSPROT_RELATIONS.get(u_ac, set()))
                    try:
                        childSwissProt = str(any(_go_ids_accept_single(inSwissProt, go) for inSwissProt in SWISSPROT_RELATIONS.get(u_ac, set())))
                    except KeyError:
                        childSwissProt = str(False)

                    cols = ["RELATION", u_ac, go, name, inSwissProt, childSwissProt, "", str(count)]
                    cols = cols + [docid for docid, _ in micro_counter[rel_key].most_common()]
                    f.write("\t".join(cols) + "\n")

            rel_evaluation = macro_counter.most_common(100)

    return rel_evaluation


def evaluate_with_argv(argv=[]):
    args = parse_arguments(argv)

    training_corpus = read_corpus(args.corpus, args.corpus_percentage, args.predict_entities)
    test_corpus = None
    if args.test_corpus:
        test_corpus = read_corpus(args.test_corpus, args.corpus_percentage, args.predict_entities)

    print_run_args(args, training_corpus)
    print()
    result = evaluate(training_corpus, test_corpus, args)
    print()
    print_run_args(args, training_corpus)
    print_corpus_pipeline_dependent_stats(training_corpus)
    print()

    return result


def read_corpus(corpus_name, corpus_percentage=1.0, predict_entities=False):
    import os
    from nalaf.utils.readers import HTMLReader
    from nalaf.utils.annotation_readers import AnnJsonAnnotationReader

    __corpora_dir = repo_path(["resources", "corpora"])

    if corpus_name in ["LocText", "LocText_v2"]:  # With reviewed normalizations (8 new protein normalizations made by Tanya; no other modifications)
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_annjson_with_normalizations_latest_5_feb_2017/')

    elif corpus_name == "LocText_v1":  # With normalizations; normalizations done in excel sheet
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_annjson_with_normalizations/')

    elif corpus_name == "LocText_v0":  # Original as annotated from tagtog, without normalizations at all
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_master_json/pool/')

    elif corpus_name in ["SwissProt", "NewDiscoveries"]:

        if corpus_name == "SwissProt":
            pmids_file_path = os.path.join(repo_path(["resources", "features", "human_localization_all_PMIDs_only__2016-11-20.tsv"]))

        elif corpus_name == "NewDiscoveries":
            pmids_file_path = os.path.join(repo_path(["tmp", "pubmed_result.txt"]))

        dir_html = None
        corpus = Dataset()

        with DownloadArticle() as PMID_DL:
            with open(pmids_file_path) as f:
                for pmid in f:
                    pmid = pmid.strip()

                    for _, doc in PMID_DL.download([pmid]):
                        try:
                            doc.get_text()  # some empty docs?
                            corpus.documents[pmid] = doc
                        except:
                            pass

    else:
        raise AssertionError(("Corpus not recognized: ", corpus_name))

    if dir_html is not None:
        corpus = HTMLReader(dir_html).read()

        if corpus_name.startswith("LocText"):
            # Remove PMCs, full-text
            del corpus.documents["PMC3596250"]
            del corpus.documents["PMC2192646"]
            del corpus.documents["PMC2483532"]
            del corpus.documents["PMC2847216"]

        AnnJsonAnnotationReader(
            dir_annjson,
            read_only_class_id=[PRO_ID, LOC_ID, ORG_ID],
            read_relations=True,
            delete_incomplete_docs=False).annotate(corpus)

    if (corpus_percentage < 1.0):
        corpus, _ = corpus.percentage_split(corpus_percentage)

    if predict_entities:
        # only human if dir_html is None == no LocText corpus, otherwise tag for the organisms that are in LocText
        tagger_entity_types = "-22,-3,9606,3702,4932" if dir_html else "-22,-3,9606"
        # print(tagger_entity_types)

        STRING_TAGGER = StringTagger(PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, tagger_entity_types=tagger_entity_types, send_whole_once=True)
        STRING_TAGGER.annotate(corpus)

    return corpus


def print_run_args(args, corpus):
    print("Train Arguments: ")
    for key, value in sorted((vars(args)).items()):
        print("\t{} = {}".format(key, value))

    print_corpus_hard_core_stats(corpus)


def print_corpus_hard_core_stats(corpus):
    print()
    print("Corpus stats:")
    print("\t#documents: {}".format(len(corpus)))
    print("\t#relations: {}".format(len(list(corpus.relations()))))


def print_corpus_pipeline_dependent_stats(corpus):

    # Assumes the sentences and edges have been generated (through relations_pipeline)

    T = 0
    P = 0
    N = 0

    for e in corpus.edges():
        T += 1
        if e.is_relation():
            P += 1
        else:
            N += 1

    # Totals for whole corpus (test data too) and with SentenceDistanceEdgeGenerator (only same sentences)
    # abstracts only -- #docs: 100 -- #P=351 vs. #N=308
    # abstract + fulltext -- #docs: 104, P=614 vs N=1480

    print("\t#sentences: {}".format(len(list(corpus.sentences()))))
    print("\t#instances (edges): {} -- #P={} vs. #N={}".format(T, P, N))
    print("\t#plausible relations from edges: {}".format(len(list(corpus.plausible_relations_from_generated_edges()))))
    print("\t#features: {}".format(next(corpus.edges()).features_vector.shape[1]))

    return (P, N)


if __name__ == "__main__":
    import sys
    ret = evaluate_with_argv(sys.argv[1:])
    print(ret)
