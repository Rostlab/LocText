from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.learning.svmlight import SVMLightTreeKernels
from loctext.learning.annotators import LocTextRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf import print_verbose, print_debug

def parse_arguments(argv=[]):
    import argparse

    parser = argparse.ArgumentParser(description='dooh')

    parser.add_argument('--corpus', default="LocText", choices=["LocText"])
    parser.add_argument('--corpus_percentage', type=float, required=True, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')
    parser.add_argument('--minority_class', type=int, default=1, choices=[-1, 1])
    parser.add_argument('--majority_class_undersampling', type=float, default=1.0, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    parser.add_argument('--svm_hyperparameter_c', type=float, default=0.0005)
    parser.add_argument('--svm_threshold', type=float, default=0)
    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)
    parser.add_argument('--use_tk', default=False, action='store_true')

    args = parser.parse_args(argv)

    return args


def parse_arguments_string(arguments=""):
    return parse_arguments(arguments.split("\s+"))


def train(training_set, args):
    # WARN: we should read the class ids from the corpus

    # feature_generators = LocTextRelationExtractor.default_feature_generators(PRO_ID, LOC_ID)
    feature_generators = None
    pipeline = RelationExtractionPipeline(PRO_ID, LOC_ID, REL_PRO_LOC_ID, feature_generators=feature_generators)

    # Learn
    pipeline.execute(training_set, train=True)
    svmlight = SVMLightTreeKernels(use_tree_kernel=args.use_tk)
    instancesfile = svmlight.create_input_file(training_set, 'train', pipeline.feature_set, minority_class=args.minority_class, majority_class_undersampling=args.majority_class_undersampling)
    svmlight.learn(instancesfile, c=args.svm_hyperparameter_c)

    annotator = LocTextRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, pipeline=pipeline, svmlight_bin_model=svmlight.model_path, svmlight=svmlight, svm_threshold=args.svm_threshold)

    return annotator.annotate


def evaluate(corpus, args):
    annotator_gen_fun = (lambda training_set: train(training_set, args))
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, evaluator, args.k_num_folds, use_validation_set=not args.use_test_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    return rel_evaluation


def evaluate_with_argv(argv=[]):
    args = parse_arguments(argv)

    if (args.corpus_percentage == 1.0):
        corpus = read_corpus(args.corpus)
    else:
        corpus, _ = read_corpus(args.corpus).percentage_split(args.corpus_percentage)

    # Print the stats twice, before and after whole pipeline, so the info does not get lost in the possible long log
    # print_stats(corpus, args)
    result = evaluate(corpus, args)
    # print_stats(corpus, args)
    print(result)

    return result


def read_corpus(corpus_name):
    import os
    from nalaf.utils.readers import HTMLReader
    from nalaf.utils.annotation_readers import AnnJsonAnnotationReader

    __corpora_dir = repo_path(["resources", "corpora"])

    if corpus_name == "LocText":
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_master_json/pool/')

    corpus = HTMLReader(dir_html).read()

    # Remove PMCs, full-text
    del corpus.documents["PMC3596250"]
    del corpus.documents["PMC2192646"]
    del corpus.documents["PMC2483532"]
    del corpus.documents["PMC2847216"]

    AnnJsonAnnotationReader(
        dir_annjson,
        read_only_class_id=None,
        read_relations=True,
        delete_incomplete_docs=False).annotate(corpus)

    return corpus


def print_run_args(args):
    print("Arguments: ")
    for key, value in sorted((vars(args)).items()):
        print("\t{} = {}".format(key, value))
    print()


def print_stats(corpus, args):
    from nalaf.preprocessing.edges import SimpleEdgeGenerator
    from nalaf.preprocessing.spliters import NLTKSplitter
    from nalaf.preprocessing.tokenizers import TmVarTokenizer

    splitter = NLTKSplitter()
    tokenizer = TmVarTokenizer()  # TODO change
    edger = SimpleEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID)

    splitter.split(corpus)
    tokenizer.tokenize(corpus)
    edger.generate(corpus)
    corpus.label_edges()

    P = 0
    N = 0

    for e in corpus.edges():
        assert e.target != 0, str(e)
        # print_verbose(e, e.target)

        if e.target > 0:
            P += 1
        else:
            N += 1

    # with all (abstract+fulltext), P=614 vs N=1480
    # with only abstracts -- Corpus size: 100 -- #P=351 vs. #N=308
    print("Corpus size: {} -- #P={} vs. #N={}".format(len(corpus), P, N))
    print_run_args(args)

    return (P, N)


if __name__ == "__main__":
    import sys
    evaluate_with_argv(sys.argv[1:])
