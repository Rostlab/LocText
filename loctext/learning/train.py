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
    parser.add_argument('--minority_class', type=int, default=1, choices=[-1, 1])
    parser.add_argument('--minority_undersampling', type=float, default=1, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    parser.add_argument('--svm_hyperparameter_c', type=float, default=0.0005)
    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)
    parser.add_argument('--use_tk', default=False, action='store_true')

    return parser.parse_args(argv)


def parse_arguments_string(arguments=""):
    return parse_arguments(arguments.split("\s+"))


def train(training_set, args):

    feature_generators = LocTextRelationExtractor.default_feature_generators(PRO_ID, LOC_ID)
    # Alert: we should read the class ids from the corpus
    pipeline = RelationExtractionPipeline(PRO_ID, LOC_ID, REL_PRO_LOC_ID, feature_generators=feature_generators)

    # Learn
    pipeline.execute(training_set, train=True)
    svmlight = SVMLightTreeKernels(use_tree_kernel=args.use_tk)
    instancesfile = svmlight.create_input_file(training_set, 'train', pipeline.feature_set, minority_class=1) # , minority_class=None, undersampling=0.1)
    svmlight.learn(instancesfile, c=0.0005)

    # Alert: we should read the class ids from the corpus
    return LocTextRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, svmlight.model_path, pipeline=pipeline, svmlight=svmlight)


def evaluate(corpus, args):
    annotator_fun = (lambda training_set: train(training_set, args))
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    evaluations = Evaluations.cross_validate(annotator_fun, corpus, evaluator, args.k_num_folds, use_validation_set=args.use_validation_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    return rel_evaluation


def evaluate_with_argv(argv=[]):
    args = parse_arguments(argv)
    corpus = read_corpus(args.corpus)

    return evaluate(corpus, args)


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
        read_relations=True,
        read_only_class_id=None,
        delete_incomplete_docs=False).annotate(corpus)

    return corpus


def print_corpus_stats(corpus):
    from nalaf.preprocessing.edges import SimpleEdgeGenerator
    from nalaf.preprocessing.spliters import NLTKSplitter
    from nalaf.preprocessing.tokenizers import TmVarTokenizer

    splitter = NLTKSplitter()
    tokenizer = TmVarTokenizer()
    edger = SimpleEdgeGenerator(PRO_ID, LOC_ID, REL_PRO_LOC_ID)

    splitter.split(corpus)
    tokenizer.tokenize(corpus)
    edger.generate(corpus)
    corpus.label_edges()

    P = 0
    N = 0

    for e in corpus.edges():
        assert e.target != 0, str(e)
        print_verbose(e, e.target)

        if e.target > 0:
            P += 1
        else:
            N += 1

    # with all (abstract+fulltext), P=614 vs N=1480
    # with only abstracts -- Corpus size: 100 -- #P=351 vs. #N=308
    print("Corpus size: {} -- #P={} vs. #N={}".format(len(corpus), P, N))

    return (P, N)


if __name__ == "__main__":
    import sys
    evaluate_with_argv(sys.argv[1:])
