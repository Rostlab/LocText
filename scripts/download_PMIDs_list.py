import sys
import urllib.request

from nalaf.utils.download import DownloadArticle

call_online_string_tagger = False

##

in_path = "resources/features/human_localization_all_PMIDs_only__2016-11-20.tsv"

with DownloadArticle() as PMID_DL:

    with open(in_path) as f:
        for pmid in f:
            pmid = pmid.strip()

            for dl_pmid, doc in PMID_DL.download([pmid]):
                print(dl_pmid)

            if call_online_string_tagger:
                url = "http://compartments.jensenlab.org/document/{}/annotations".format(pmid)
                out_path = "resources/features/human_localization_all_PMIDs_only_StringTagger_results__2016-11-20/{}.json".format(pmid)
                print("\t", url)
                urllib.request.urlretrieve(url, out_path)
