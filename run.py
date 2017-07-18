import pickle
from nalaf.utils.download import DownloadArticle
from nalaf.utils.readers import StringReader, PMIDReader
from loctext.learning.annotators import StringTagger
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, repo_path
from loctext.learning.annotators import LocTextDXModelRelationExtractor


RE_MODEL_PATH = repo_path("resources", "models", "D0_9606,3702,4932_1497520729.163767.bin")
RE_FEATURES_PATH = repo_path("resources", "features", "selected", "0_True_LinearSVC_C=2.0-1487943476.673364-NAMES.py")
RE_MODEL_BIN = None
with open(RE_MODEL_PATH, "rb") as f:
    RE_MODEL_BIN = pickle.load(f)


def parse_arguments(argv=[]):

    import argparse

    parser = argparse.ArgumentParser(description='Run LocText on some text to extract Protein<-->Cell Compartments relations')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text",
                             help="Run against given text/string")
    input_group.add_argument("--pmid",
                             help="Run against the abstract of the given PubMed ID (PMID), downloaded from NCBI")

    input_group.add_argument("--entity_tagger_url",
                             default="http://127.0.0.1:5000",
                             help="URL (include host and port) of the dockerized STRING Tagger server")

    args = parser.parse_args()

    return args


def read_models(args):
    # Note, the id constants for the entities and relations (*_ID) are arbitrary. Nonetheless, you must know them to
    # later extract your desired types of entities/relations

    ner = StringTagger(PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, host=args.entity_tagger_url)

    re = LocTextDXModelRelationExtractor(
        PRO_ID, LOC_ID, REL_PRO_LOC_ID,
        sentence_distance=0,
        selected_features_file=RE_FEATURES_PATH,
        use_predicted_entities=True,
        model=RE_MODEL_BIN,
        #
        preprocess=True,
        #
        class_weight=None,
        kernel='linear',
        C=1,
    )

    return (ner, re)


def run_with_argv(argv=[]):
    args = parse_arguments(argv)

    ner, re = read_models(args)

    if args.text:
        corpus = StringReader(args.text).read()
    elif args.pmid:
        corpus = PMIDReader(args.pmid).read()
    # See more possible readers including some NCBI XML files in `nalaf.utils.readers`

    ner.annotate(corpus)
    re.annotate(corpus)

    return corpus


if __name__ == "__main__":
    import sys
    annotated_corpus = run_with_argv(sys.argv[1:])

    print()
    print("# Predicted entities:")
    for entity in annotated_corpus.predicted_entities():
        print(entity)

    print()
    print("# Predicted relations:")
    for relation in annotated_corpus.predicted_relations():
        print(relation)
