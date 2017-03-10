import pickle
from collections import Counter
import math
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, GO_NORM_ID, repo_path
import re

in_path = "resources/features/SwissProt_all_corpus_organisms_relations__2017-02-15.tsv"
# get all uniprot identifiers: cat <...>.tsv |  awk '{print $1}' | sort | uniq | wc

regex_go_id = re.compile('GO:[0-9]+')

with open(in_path) as f:
    next(f)  # skip header

    all_relations = {}

    for line in f:
        upid, organism_id, localization_gos = line.split("\t")
        organism_id = int(organism_id)

        organism_relations = all_relations.get(organism_id, {})
        go_terms = regex_go_id.findall(localization_gos)
        organism_relations[upid] = set(go_terms)
        all_relations[organism_id] = organism_relations

    print("Total uniprot entries:", sum(len(organism_relations) for organism_relations in all_relations.values()))
    print("9606 unit prot entiries:", (len(all_relations[9606])))

    out_path = repo_path("resources", "features", "SwissProt_all_relations.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(all_relations, f)
