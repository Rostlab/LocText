import pickle
from collections import Counter
import math
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, GO_NORM_ID, repo_path
import glob
from loctext.util import simple_parse_GO
import json
from pprint import pprint
import re

# Run grep -o "GO:[0-9]*" "resources/features/human_localization_all__2016-11-20.tsv" | sort | uniq -c | sort > scripts/precomputed_SwissProt_normalized_total_absolute_loc_rels_ratio.txt

in_path = "scripts/precomputed_SwissProt_normalized_total_absolute_loc_rels_ratio.txt"

mode = "normalized"

regex_canonical_go_id = re.compile("canonical[\s\W]*?(GO:[0-9]+)")

with open(in_path) as f:
    counter_relations = Counter({go: int(count) for line in f for count, go in [line.strip().split(" ")]})

GO_TREE = simple_parse_GO.simple_parse(repo_path(["resources", "ontologies", "go-basic.cellular_component.latest.obo"]))

counter_mentions = Counter()

for json_path in glob.glob("resources/features/human_localization_all_PMIDs_only_StringTagger_results__2016-11-20/*.json"):
    with open(json_path) as f:
        # We read the file as string since we saw json-parsing errors or inconsistencies
        data = f.read()

        for go in regex_canonical_go_id.findall(data):
            if go in GO_TREE:
                counter_mentions.update([go])

for key in counter_relations:
    if key not in counter_mentions:
        counter_mentions.update({key: 1})

for key in counter_mentions:
    if key not in counter_relations:
        counter_relations.update({key: 0})

ret = {}

for key, mention_count in counter_mentions.items():
    relation_count = counter_relations[key]
    ret[key] = (relation_count / mention_count)

out_path = repo_path(["resources", "features", "SwissProt_normalized_total_background_loc_rels_ratios.pickle"])
with open(out_path, "wb") as f:
    pickle.dump(ret, f)

print(":-)")
