from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from nalaf.features.stemming import ENGLISH_STEMMER
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from util import my_cv_generator
import time
from collections import Counter
import pickle

corpus = read_corpus("LocText")

mode = "unnormalized"

mention_mode_total_counter = {key: Counter() for key in [PRO_ID, LOC_ID, ORG_ID]}
relation_mode_total_counter = {key: Counter() for key in [PRO_ID, LOC_ID, ORG_ID]}
ratio_mode_total_counter = {key: Counter() for key in [PRO_ID, LOC_ID, ORG_ID]}

#

def entity2text(entity):
    return ENGLISH_STEMMER.stem(entity.text)

#

for entity in corpus.entities():
    mention_mode_total_counter[entity.class_id].update({entity2text(entity)})


for rel in corpus.relations():
    if rel.class_id != REL_PRO_LOC_ID:
        continue

    relation_mode_total_counter[rel.entity1.class_id].update({entity2text(rel.entity1)})
    relation_mode_total_counter[rel.entity2.class_id].update({entity2text(rel.entity2)})


for type_key, mention_counter in mention_mode_total_counter.items():
    relation_counter = relation_mode_total_counter[type_key]

    for entity_key, mention_count in mention_counter.items():
        relation_count = relation_counter.get(entity_key, 0)
        ratio_mode_total_counter[type_key][entity_key] = (relation_count / mention_count)

#######################################################################################################################

# for ratio_counter in ratio_mode_total_counter.values():
#     ratio_sorted = sorted(ratio_counter.items(), key=lambda pair: pair[1])
#
#     print()
#     for key, val in ratio_sorted:
#         print('"{}": {},'.format(key, val))
#
#     print()
#     print()
#
# print()

out_path = repo_path(["resources", "features", "corpus_" + mode + "_total_absolute_loc_rels_ratios.pickle"])
with open(out_path, "wb") as f:
    pickle.dump(ratio_mode_total_counter[LOC_ID], f)

for marker in {"GFP", "RFP", "CYH2", "ALG2", "MSB2", "KSS1", "KRE11", "SER2", "Snf7"}:
    print(marker, ratio_mode_total_counter[PRO_ID].get(ENGLISH_STEMMER.stem(marker)))
