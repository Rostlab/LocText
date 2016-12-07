import os

PRO_ID = 'e_1'
LOC_ID = 'e_2'
ORG_ID = 'e_3'
REL_PRO_LOC_ID = 'r_5'

UNIPROT_NORM_ID = 'n_7'
GO_NORM_ID = 'n_8'
TAXONOMY_NORM_ID = 'n_9'
STRING_NORM_ID = 'n_10'

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def repo_path(listOrString):
    if type(listOrString) is str:
        listOrString = [listOrString]

    return os.path.join(repo_root, *listOrString)
