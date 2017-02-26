import pickle
from loctext.util import repo_path
from loctext.learning.evaluations import relation_accept_uniprot_go, are_go_parent_and_child, get_localization_name


with open(repo_path("resources", "features", "SwissProt_relations.pickle"), "rb") as f:
    SWISSPROT_RELATIONS = pickle.load(f)


def is_in_swiss_prot(uniprot_ac, go):
    explicitly_written = go in SWISSPROT_RELATIONS.get(uniprot_ac, set())

    return explicitly_written or is_parent_of_swiss_prot_annotation(uniprot_ac, go)


def is_parent_of_swiss_prot_annotation(uniprot_ac, go):
    try:
        return any(are_go_parent_and_child(go, swiss_prot_go) for swiss_prot_go in SWISSPROT_RELATIONS.get(uniprot_ac, set()))
    except KeyError:
        return False


def is_child_of_swiss_prot_annotation(uniprot_ac, go):
    try:
        return any(are_go_parent_and_child(swiss_prot_go, go) for swiss_prot_go in SWISSPROT_RELATIONS.get(uniprot_ac, set()))
    except KeyError:
        return False


def parse(filepath, previous_annotations=None):
    ret = {}
    with open(filepath) as f:
        next(f)  # discard header
        for line in f:
            line = line.strip()
            typ, uniprot_ac, go, loc_name, inSwissProt, childSwissProt, confirmed, num_docs, *docs = line.split("\t")

            if previous_annotations is None:
                if not confirmed:
                    break
            else:
                confirmed = any(previous.get(uniprot_ac, set()) == go for previous in previous_annotations)
                inSwissProt = is_in_swiss_prot(uniprot_ac, go)
                childSwissProt = childSwissProt
                row = [typ, uniprot_ac, go, loc_name, inSwissProt, childSwissProt, confirmed, num_docs] + docs
                print('\t'.join(rows))
    return ret


if __name__ == "__main__":
    import sys
    main_filepath = sys.argv[1]
    previous_results_filepaths = sys.argv[2:]
    assert len(previous_results_filepaths) == 2
    previous_annotations = [parse(filepath, None) for filepath in previous_results_filepaths]

    parse(filepath, previous_annotations)
