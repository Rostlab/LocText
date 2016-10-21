from loctext.util import PRO_ID, LOC_ID, REL_PRO_LOC_ID, repo_path
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations

def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(description='dooh')

    parser.add_argument('--corpus', default="LocText", choices=["LocText"])
    parser.add_argument('--use_tk', default=False, action='store_true')
    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)

    return parser.parse_args()

def train(argv):
    args = parse_arguments(argv)
    corpus = read_corpus(args.corpus)

    annotator = StubSameSentenceRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID)
    evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, match_case=False)

    ret = Evaluations.cross_validate(annotator, corpus, evaluator, args.k_num_folds, use_validation_set=(not args.use_test_set))

    print(ret)


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
