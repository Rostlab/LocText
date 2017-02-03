from nalaf.learning.lib.sklsvm import SklSVM
from nalaf.structures.data import Dataset
from nalaf.features.stemming import ENGLISH_STEMMER
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor
from util import my_cv_generator
import time
from collections import Counter

corpus = read_corpus("LocText")
# locTextModel = LocTextSSmodelRelationExtractor(PRO_ID, LOC_ID, REL_PRO_LOC_ID)
# locTextModel.pipeline.execute(corpus, train=True)

pro_all_counter = Counter()
loc_all_counter = Counter()

pro_is_related_counter = Counter()
loc_is_related_counter = Counter()

pro_ratio = {}
loc_ratio = {}

def entity2text(entity):
    return ENGLISH_STEMMER.stem(entity.text)

for entity in corpus.entities():
    if entity.class_id == PRO_ID:
        pro_all_counter.update({entity2text(entity)})
    elif entity.class_id == LOC_ID:
        loc_all_counter.update({entity2text(entity)})

for rel in corpus.relations():
    if rel.class_id != REL_PRO_LOC_ID:
        continue

    if rel.entity1.class_id == PRO_ID:
        pro, loc = rel.entity1, rel.entity2
    else:
        pro, loc = rel.entity2, rel.entity1

    pro_is_related_counter.update({entity2text(pro)})
    loc_is_related_counter.update({entity2text(loc)})

for pro, total_count in pro_all_counter.items():
    rel_count = pro_is_related_counter.get(pro, 0)
    pro_ratio[pro] = (rel_count / total_count)

for loc, total_count in loc_all_counter.items():
    rel_count = loc_is_related_counter.get(loc, 0)
    loc_ratio[loc] = (rel_count / total_count)

pro_ratio_sorted = sorted(pro_ratio.items(), key=lambda pair: pair[1])
loc_ratio_sorted = sorted(loc_ratio.items(), key=lambda pair: pair[1])

#######################################################################################################################

# print()
# print(pro_all_counter)
# print()
# print(loc_all_counter)
# print()
# print(pro_is_related_counter)
# print()
# print(loc_is_related_counter)
print()
for key, val in pro_ratio_sorted:
    print('"{}": {},'.format(key, val))

print()
print()

for key, val in loc_ratio_sorted:
    print('"{}": {},'.format(key, val))

print()

for marker in {"GFP", "RFP", "CYH2", "ALG2", "MSB2", "KSS1", "KRE11", "SER2", "Snf7"}:
    print(marker, pro_ratio.get(ENGLISH_STEMMER.stem(marker)))
