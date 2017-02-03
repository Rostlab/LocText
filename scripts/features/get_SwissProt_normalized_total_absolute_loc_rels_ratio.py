import pickle
from collections import Counter
import math
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, GO_NORM_ID, repo_path

# Run grep -o "GO:[0-9]*" "resources/features/human_localization_all__2016-11-20.tsv" | sort | uniq -c | sort > scripts/precomputed_SwissProt_normalized_total_absolute_loc_rels_ratio.txt

in_path = "scripts/precomputed_SwissProt_normalized_total_absolute_loc_rels_ratio.txt"

SCALING_TYPE = "log"  # or "log"

with open(in_path) as f:

    counter = Counter({go: int(count) for line in f for count, go in [line.strip().split(" ")]})
    max_go, maximum_count = counter.most_common(n=1)[0]

    if SCALING_TYPE == "abs":
        scale = (lambda count: count / maximum_count)
    elif SCALING_TYPE == "log":
        scale = (lambda count: math.log(count, maximum_count))

    ret = {go: scale(count) for go, count in counter.items()}

    print()
    print("Total count: ", len(ret))
    print("Top counts:")

    for top, _ in counter.most_common(n=30):
        print("\t", top, counter[top], " --> ", ret[top])

    out_path = repo_path(["resources", "features", "SwissProt_normalized_total_absolute_loc_rels_ratios.pickle"])

    with open(out_path, "wb") as f:
        pickle.dump(ret, f)
