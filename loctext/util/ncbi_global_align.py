#!/usr/bin/env python3
import sys
import os
import re
try:
    import requests
    import requests_cache
except Exception as e:
    print("ERROR", "- You need to install missing dependencies: pip install requests requests-cache\n")
    raise

assert sys.version_info.major == 3, "the script requires Python 3"

__author__ = "Juan Miguel Cejuela (@juanmirocks)"

__help__ = """
            A bit of a hack that uses the NCBI Global Alignment API to align two proteins (Needleman-Wunsch algorithm)
            Also (optionally) parse out a column of the tabular output, e.g. column 2 == sequence identity percentage.
            Http requests are cached; a cache file is saved into the user's home folder.

            Example call: ./ncbi_global_align.py P08100 P02699 2

            See: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp&BLAST_PROGRAMS=blastp&PAGE_TYPE=BlastSearch&BLAST_SPEC=GlobalAln&LINK_LOC=blasttab&LAST_PAGE=blastn&BLAST_INIT=GlobalAln
            Api doc: https://ncbi.github.io/blast-cloud/dev/api.html

            WARNING: this script uses options that are not documented in the API; will likely break. Tested as of: 2017-04-14
           """

# ----------------------------------------------------------------------------


CACHE_FILE_NAME = os.path.join(os.path.expanduser('~'), "TMP_CACHE_NCBI_GLOBAL_ALIGN")  # home folder

requests_cache.install_cache(
    cache_name=CACHE_FILE_NAME,
    backend="sqlite",
    expire_after=(24 * 60 * 60),  # 1 day, just like the NCBI expiration date
    allowable_methods=('GET', "POST"))


# ----------------------------------------------------------------------------


POST_URL = "https://blast.ncbi.nlm.nih.gov/BlastAlign.cgi?CMD=Put&PROGRAM=blastp&BLAST_SPEC=GlobalAln"
GET_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Get&FORMAT_TYPE=Tabular"


def post(seq1, seq2):
    params = {}
    params["QUERY"] = seq1
    params["SUBJECTS"] = seq2

    response = requests.post(POST_URL, params=params)
    assert response.ok, response

    rid_search = re.search('RID = (\\S+)', response.text)

    assert rid_search, "No RID found (job id / result id)"
    rid = rid_search.group(1)
    return rid


def get(rid, column=None):
    params = {}
    params["RID"] = rid

    response = requests.get(GET_URL, params=params)
    assert response.ok, response

    real_body = ""
    in_pre = False
    for line in response.text.splitlines():
        if line.lower() == "<pre>":
            in_pre = True
        elif line.lower() == "</pre>":
            break
        elif in_pre and not line.startswith("#"):
            real_body += line

    try:
        assert len(real_body) > 0
        if column is None:
            return real_body
        else:
            return real_body.split("\t")[column]

    except Exception as e:
        raise AssertionError(("No valid response output", response.text), e)




def global_align(seq1, seq2, column=None):
    rid = post(seq1, seq2)
    return get(rid, column)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        assert len(sys.argv) in {3, 4}
        column = None if len(sys.argv) == 3 else int(sys.argv[3])

        ret = global_align(seq1=sys.argv[1], seq2=sys.argv[2], column=column)

        print(ret)

    except Exception:
        print(__help__)
        print()
        raise
