from loctext.learning.evaluations import is_in_swiss_prot, is_child_of_swiss_prot_annotation, accept_relation_uniprot_go, are_go_parent_and_child, get_localization_name

def parse(filepath, previous_annotations=None):
    ret = {}
    with open(filepath) as f:
        header = next(f)
        if previous_annotations:
            print(header.strip())

        for line in f:
            line = line.strip()
            typ, uniprot_ac, go, loc_name, inSwissProt, childSwissProt, confirmed, num_docs, *docs = line.split("\t")

            rel_key = uniprot_ac, go

            if previous_annotations is None:
                if not confirmed:
                    break
                else:
                    ret[rel_key] = confirmed
            else:
                confirmed = previous_annotations.get(rel_key, "")
                inSwissProt = str(is_in_swiss_prot(uniprot_ac, go))
                childSwissProt = childSwissProt
                row = [typ, uniprot_ac, go, loc_name, inSwissProt, childSwissProt, confirmed, num_docs] + docs
                print('\t'.join(row))
    return ret


if __name__ == "__main__":
    import sys

    main_filepath = sys.argv[1]

    # Order matters
    previous_results_filepaths = sys.argv[2:]
    assert len(previous_results_filepaths) == 2
    previous_annotations = {}
    for filepath in previous_results_filepaths:
        previous_annotations.update(parse(filepath, None))

    parse(main_filepath, previous_annotations)
