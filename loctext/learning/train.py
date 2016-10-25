from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.learning.svmlight import SVMLightTreeKernels
from loctext.learning.annotators import LocTextRelationExtractor

def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(description='dooh')

    parser.add_argument('--corpus', default="LocText", choices=["LocText"])
    parser.add_argument('--use_tk', default=False, action='store_true')
    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)

    return parser.parse_args()


def train_with_argv(argv):
    args = parse_arguments(argv)
    corpus = read_corpus(args.corpus)

    return train(corpus, args)


def train(training_set, args):

    feature_generators = LocTextRelationExtractor.default_feature_generators(PRO_ID, LOC_ID)
    # Alert: we should read the class ids from the corpus
    pipeline = RelationExtractionPipeline(PRO_ID, LOC_ID, REL_PRO_LOC_ID, feature_generators=feature_generators)

    # Learn
    pipeline.execute(training_set, train=True)
    svmlight = SVMLightTreeKernels(use_tree_kernel=False)  # Beware: should use args, but conflict of Namespace vs object
    instancesfile = svmlight.create_input_file(training_set, 'train', pipeline.feature_set)
    svmlight.learn(instancesfile)

    # Alert: we should read the class ids from the corpus
    return LocTextRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID, svmlight.model_path, pipeline=pipeline, svmlight=svmlight)


def read_corpus(corpus_name):
    import os
    from nalaf.utils.readers import HTMLReader
    from nalaf.utils.annotation_readers import AnnJsonAnnotationReader

    __corpora_dir = repo_path(["resources", "corpora"])

    if corpus_name == "LocText":
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_master_json/pool/')

    corpus = HTMLReader(dir_html).read()
    AnnJsonAnnotationReader(
        dir_annjson,
        read_relations=True,
        read_only_class_id=None,
        delete_incomplete_docs=False).annotate(corpus)

    return corpus


if __name__ == "__main__":
    import sys
    train(sys.argv[1:])
