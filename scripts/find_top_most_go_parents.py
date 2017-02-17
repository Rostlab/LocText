from loctext.util import simple_parse_GO
from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, repo_path
from loctext.learning.evaluations import are_go_parent_and_child

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
    "GO:0016021",
    "GO:0009507",
    "GO:0009535",
    "GO:0005737",
    "GO:0005783",
    "GO:0005789",
    "GO:0005576",
    "GO:0005794",
    "GO:0000139",
    "GO:0005743",
    "GO:0005739",
    "GO:0031965",
    "GO:0005634",
    "GO:0005778",
    "GO:0005777",
    "GO:0005886",
    "GO:0009536",
    "GO:0005774",
    "GO:0005773",
]

Juanmi = [
    "GO:0016020"
]

whole_set = set(Lars + Tanya + Juanmi)
final_set = set()

for item in whole_set:
    if not any(are_go_parent_and_child(x, item) for x in whole_set if x != item):
        final_set.update({item})

subcellular_components_top_parents = {
    'GO:0016020',  # membrane -- http://amigo.geneontology.org/amigo/term/GO:0016020
    'GO:0005576',  # extracellular region
    'GO:0005737',  # cytoplasm
    'GO:0005856',  # cytoskeleton
    'GO:0005634',  # nucleus
}

print(final_set)
