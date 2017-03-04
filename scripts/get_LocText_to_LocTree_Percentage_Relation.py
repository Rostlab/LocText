import re
import json
import sys
from loctext.learning.evaluations import are_go_parent_and_child

regex_go_id = re.compile('GO:[0-9]+')


def get_loctree_relation_records(file_path):
    with open(file_path) as f:
        next(f)
        relations = {}

        # A record from LocTree is shown below
        # sp|P48200|IREB2_HUMAN	100	cytoplasm	cytoplasm GO:0005737(NAS); cytosol GO:0005829(IEA); mitochondrion GO:0005739(IEA);
        for line in f:

            protein_id, score, localization, gene_ontology_terms = line.split("\t")

            protein_id = protein_id.split("|")[1]
            go_terms = regex_go_id.findall(gene_ontology_terms)
            relations[protein_id] = set(go_terms)

        return relations


def get_loctext_relation_records(loctree_relations, file_path, swissprot_says, min_num_docs):
    with open(file_path) as f:
        next(f)
        positive_relations = {}
        negative_relations = {}
        # A record from LocText is shown below with column numbers
        #     0        1        2           3         4       5    6    7       8
        # RELATION	Q9H869	GO:0005737	cytoplasm	True	True		8	27979971
        for line in f:
            record = line.split("\t")

            if record[4].startswith(swissprot_says) and int(record[7]) >= min_num_docs:
                loctext_uniprot_ac = record[1]
                loctext_go_id = record[2]
                rel_key = (loctext_uniprot_ac, loctext_go_id)
                loctree_go_ids = loctree_relations.get(loctext_uniprot_ac, {})

                try:
                    is_positive_a = any(are_go_parent_and_child(loctext_go_id, loctree_go_id) for loctree_go_id in loctree_go_ids)  # loctree_go_id in loctext_go_ids
                    is_positive_b = any(are_go_parent_and_child(loctree_go_id, loctext_go_id) for loctree_go_id in loctree_go_ids)  # loctext_go_id in loctree_go_ids
                    is_positive = is_positive_a or is_positive_b
                except:
                    # print("ONE IS DEPRECATED", loctree_go_ids)
                    is_positive = False

                if is_positive:
                    positive_relations[rel_key] = loctext_go_id
                else:
                    negative_relations[rel_key] = loctext_go_id

        return positive_relations, negative_relations


if __name__ == "__main__":

    loctree_records_file_path = sys.argv[1]
    loctext_records_file_path = sys.argv[2]
    swissprot_says = sys.argv[3]  # Values 'False', True, or '' to accept any
    min_num_docs = int(sys.argv[4])

    loctree_relations = get_loctree_relation_records(loctree_records_file_path)
    positive_relations, negative_relations = get_loctext_relation_records(loctree_relations, loctext_records_file_path, swissprot_says, min_num_docs)

    positive_records = len(positive_relations)
    total_records = len(positive_relations) + len(negative_relations)

    print("***************************************************************************************************")
    print("LocText predicted RELATION ['in SwissProt' set to False], but present in LocTree: ", positive_records)
    print("Total predicted RELATION ['in SwissProt' set to False]: ", total_records)
    print("Percentage predicts of LocText to LocTree: ", (positive_records/total_records)*100)
    print("***************************************************************************************************")
