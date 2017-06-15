import sys
import urllib.request

from nalaf.utils.download import DownloadArticle

in_path = sys.argv[1]
call_online_string_tagger = False  # bool(sys.argv[2])

print(in_path, call_online_string_tagger)

##

with DownloadArticle() as PMID_DL:

    with open(in_path) as f:
        for index, pmid in enumerate(f):
            pmid = pmid.strip()

            for dl_pmid, doc in PMID_DL.download([pmid]):
                print(index, dl_pmid)

            if call_online_string_tagger:
                url = "http://compartments.jensenlab.org/document/{}/annotations".format(pmid)
                out_path = "resources/features/human_localization_all_PMIDs_only_StringTagger_results__2016-11-20/{}.json".format(pmid)
                print("\t", url)
                urllib.request.urlretrieve(url, out_path)
