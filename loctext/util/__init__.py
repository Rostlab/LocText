import os

PRO_ID = 'e_1'
LOC_ID = 'e_2'
#ORG_ID = 'e_3'
REL_PRO_LOC_ID = 'r_5'

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def repo_path(listOrString):
    if type(listOrString) is str:
        listOrString = [listOrString]

    return os.path.join(repo_root, *listOrString)
