from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, repo_path
from loctext.learning.evaluations import are_go_parent_and_child
from loctext.util import simple_parse_GO

GO_TREE = simple_parse_GO.simple_parse(repo_path(["resources", "ontologies", "go-basic.cellular_component.latest.obo"]))

Lars = [
    "GO:0005576",  # extracellular
    "GO:0005634",  # nucleus
    "GO:0005739",  # mitochondrion
    "GO:0005764",  # lysosome
    "GO:0005768",  # endosome
    "GO:0005773",  # vacuole
    "GO:0005777",  # peroxisome
    "GO:0005783",  # endoplasmic reticulum
    "GO:0005794",  # golgi apparatus
    "GO:0005829",  # cytosol
    "GO:0005856",  # cytoskeleton
    "GO:0005886",  # plasma membrane
    "GO:0009507",  # chloroplast
]

Tanya = [
    # "GO:0016021",  # integral to membrane
    "GO:0009507",  # chloroplast
    "GO:0009535",  # chloroplast thylakoid membrane
    # "GO:0005737",  # cytoplasm
    "GO:0005783",  # endoplasmic reticulum
    "GO:0005789",  # endoplasmic reticulum membrane
    "GO:0005576",  # extracellular region
    "GO:0005794",  # Golgi apparatus
    "GO:0000139",  # Golgi membrane
    "GO:0005743",  # mitochondrial inner membrane
    "GO:0005739",  # mitochondrion
    "GO:0031965",  # nuclear membrane
    "GO:0005634",  # nucleus
    "GO:0005778",  # peroxisomal membrane
    "GO:0005777",  # peroxisome
    "GO:0005886",  # plasma membrane
    "GO:0009536",  # plastid
    "GO:0005774",  # vacuolar membrane
    "GO:0005773",  # vacuole
]

Juanmi = [
    # "GO:0016020"  # membrane
    "GO:0009579",  # thylakoid  https://en.wikipedia.org/wiki/Thylakoid
    # "GO:0005840",  # ribosome
    "GO:0005618",  # cell wall
]

# -------

LocText_Extra = [
    "GO:0005929",  # cilium
    "GO:0000781",  # chromosome, telomeric region
    "GO:0005933",  # cellular bud
    "GO:0000777",  # condensed chromosome kinetochore
    "GO:0042470",  # melanosome
    "GO:0005694",  # chromosome
    "GO:0005737",  # cytoplasm
    "GO:0016021",  # integral component of membrane
    "GO:0042995",  # cell projection
    "GO:0045121",  # membrane raft
    "GO:0009986",  # cell surface
    "GO:0033107",  # CVT vesicle
    "GO:0000775",  # chromosome, centromeric region
    "GO:0005811",  # lipid particle
    "GO:0016020",  # membrane
    "GO:0030136",  # clathrin-coated vesicle
    "GO:0045202",  # synapse
    "GO:0071944",  # cell periphery
]

# -------

difficult_cases = [
    "GO:0071159",
    "GO:1990204",
    "GO:0071141",
    "GO:0034703",
    "GO:0005942",
    "GO:0016021",
]

# -------

whole_set = set(Lars + Tanya + Juanmi + LocText_Extra)
final_set = set()

for item in whole_set:
    if not any(are_go_parent_and_child(x, item) for x in whole_set if x != item):
        final_set.update({item})

# -------

print()

for difficult in difficult_cases:
    try:
        assert not any(are_go_parent_and_child(x, difficult) for x in final_set)
    except:
        print("Difficult:", difficult)

# -------

print()

print("{")
for index, item in enumerate(final_set):
    print("    '{}',  # {}: {}".format(item, index+1, GO_TREE[item].name))
print("}")
