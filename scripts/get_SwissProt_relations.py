import pickle
from collections import Counter
import math
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, GO_NORM_ID, repo_path
import re

# in_path = "resources/features/human_localization_all__2016-11-20.tsv"
in_path = "resources/features/SwissProt_all_corpus_organisms_relations__2017-02-04.tsv"
# get all uniprot identifiers: cat resources/features/SwissProt_all_corpus_organisms_relations__2017-02-04 |  awk '{print $1}' | sort | uniq | wc

regex_go_id = re.compile('GO:[0-9]+')

with open(in_path) as f:
    next(f)  # skip header

    relations = {}

    for line in f:
        # upid, _, protein_names, gene_names, localization_names, localization_gos = line.split("\t")
        upid, organism_id, localization_gos = line.split("\t")

        if organism_id in ["9606"]:
            go_terms = regex_go_id.findall(localization_gos)
            relations[upid] = set(go_terms)


    print("Total uniprot entries: " + str(len(relations)))
    # for upid, gos in relations.items():
    #     print(upid, gos)

    out_path = repo_path(["resources", "features", "SwissProt_relations.pickle"])
    with open(out_path, "wb") as f:
        pickle.dump(relations, f)
