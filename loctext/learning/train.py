from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor, LocTextDSmodelRelationExtractor, LocTextCombinedModelRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf import print_verbose, print_debug
from loctext.learning.evaluations import relation_accept_uniprot_go
from nalaf.learning.lib.sklsvm import SklSVM

def parse_arguments(argv=[]):
    import argparse

    parser = argparse.ArgumentParser(description='dooh')

    parser.add_argument('--model', required=True, choices=["SS", "DS", "Combined"])

    parser.add_argument('--corpus', default="LocText", choices=["LocText"])
    parser.add_argument('--corpus_percentage', type=float, required=True, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')
    parser.add_argument('--evaluation_level', type=int, choices=[1, 2, 3, 4], required=True)

    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)

    parser.add_argument('--feature_generators', default='LocText', choices=["LocText", "default"])
    parser.add_argument('--use_tk', default=False, action='store_true')

    parser.add_argument('--minority_class_ss_model', type=int, default=+1, choices=[-1, +1])
    parser.add_argument('--majority_class_undersampling_ss_model', type=float, default=0.9, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    parser.add_argument('--svm_hyperparameter_c_ss_model', action="store", default=0.0080)
    parser.add_argument('--svm_threshold_ss_model', type=float, default=0.0)

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
        ENTITY_MAP_FUN = 'normalized_first'
        RELATION_ACCEPT_FUN = str.__eq__
    elif args.evaluation_level == 4:
        ENTITY_MAP_FUN = 'normalized_first'
        RELATION_ACCEPT_FUN = relation_accept_uniprot_go

    args.evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, entity_map_fun=ENTITY_MAP_FUN, relation_accept_fun=RELATION_ACCEPT_FUN)

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


def _select_annotator_model(args):
    # WARN: we should read the class ids from the corpus
    pro_id = PRO_ID
    loc_id = LOC_ID
    rel_id = REL_PRO_LOC_ID

    indirect_feature_generators = {
        "LocText": None,  # Uses annotator's default
        "default": []  # Uses RelationExtractionPipeline's default

    }.get(args.feature_generators)

    ann_switcher = {
        # TODO evaluate them lazily
        "SS": LocTextSSmodelRelationExtractor(pro_id, loc_id, rel_id, feature_generators=indirect_feature_generators, model=None, classification_threshold=args.svm_threshold_ss_model, use_tree_kernel=args.use_tk),
        "DS": LocTextDSmodelRelationExtractor(pro_id, loc_id, rel_id, feature_generators=indirect_feature_generators, model=None, classification_threshold=args.svm_threshold_ds_model, use_tree_kernel=args.use_tk)
    }

    if args.model == "Combined":
        ann_switcher["Combined"] = LocTextCombinedModelRelationExtractor(pro_id, loc_id, rel_id, ss_model=ann_switcher["SS"], ds_model=ann_switcher["DS"])

    ret = ann_switcher[args.model]

    return ret


def _select_submodel_params(annotator, args):

    if isinstance(annotator, LocTextSSmodelRelationExtractor):
        return (args.minority_class_ss_model, args.majority_class_undersampling_ss_model, args.svm_hyperparameter_c_ss_model)

    elif isinstance(annotator, LocTextDSmodelRelationExtractor):
        return (args.minority_class_ds_model, args.majority_class_undersampling_ds_model, args.svm_hyperparameter_c_ds_model)

    raise AssertionError()


def train(training_set, args):

    annotator_model = _select_annotator_model(args)

    # Simple switch for either single or combined models
    submodels = annotator_model.submodels if hasattr(annotator_model, 'submodels') else [annotator_model]

    for index, annotator in enumerate(submodels):
        print("About to train model {}={}".format(index, annotator.__class__.__name__))

        annotator.pipeline.execute(training_set, train=True)

        minority_class, majority_class_undersampling, svm_hyperparameter_c = _select_submodel_params(annotator, args)

        annotator.model.train(training_set, annotator.pipeline.feature_set)

    return annotator_model.annotate


def evaluate(corpus, args):
    annotator_gen_fun = (lambda training_set: train(training_set, args))
    evaluator = args.evaluator

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, evaluator, args.k_num_folds, use_validation_set=not args.use_test_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    return rel_evaluation


def evaluate_with_argv(argv=[]):
    args = parse_arguments(argv)

    corpus = read_corpus(args.corpus, args.corpus_percentage)

    print_run_args(args, corpus)
    print()
    result = evaluate(corpus, args)
    print()
    print_run_args(args, corpus)
    print_corpus_pipeline_dependent_stats(corpus)
    print()

    return result


def read_corpus(corpus_name, corpus_percentage=1.0):
    import os
    from nalaf.utils.readers import HTMLReader
    from nalaf.utils.annotation_readers import AnnJsonAnnotationReader

    __corpora_dir = repo_path(["resources", "corpora"])

    if corpus_name == "LocText":
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_annjson_with_normalizations/')

    if corpus_name == "LocText_original":
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_master_json/pool/')

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

    return corpus


def print_run_args(args, corpus):
    print("Train Arguments: ")
    for key, value in sorted((vars(args)).items()):
        print("\t{} = {}".format(key, value))

    print_corpus_hard_core_stats(corpus)


def print_corpus_hard_core_stats(corpus):

    print("\nCorpus stats; #docs={} -- #rels={}".format(len(corpus), len(list(corpus.relations()))))


def print_corpus_pipeline_dependent_stats(corpus):

    # Assumes the sentences and edges have been generated (through relations_pipeline)

    P = 0
    N = 0

    for e in corpus.edges():
        if e.is_relation():
            P += 1
        else:
            N += 1

    # Totals for whole corpus (test data too) and with SentenceDistanceEdgeGenerator (only same sentences)
    # abstracts only -- #docs: 100 -- #P=351 vs. #N=308
    # abstract + fulltext -- #docs: 104, P=614 vs N=1480

    print("\t#sentences={}".format(len(list(corpus.sentences()))))
    print("\tedges: #P={} vs. #N={}".format(P, N))

    return (P, N)


if __name__ == "__main__":
    import sys
    ret = evaluate_with_argv(sys.argv[1:])
    print(ret)
