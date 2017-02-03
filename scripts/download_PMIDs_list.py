import urllib.request

in_path = "resources/features/human_localization_all_PMIDs_only__2016-11-20.tsv"

with open(in_path) as f:
    for pmid in f:
        pmid = pmid.strip()

        url = "http://compartments.jensenlab.org/document/{}/annotations".format(pmid)
        out_path = "resources/features/human_localization_all_PMIDs_only_StringTagger_results__2016-11-20/{}.json".format(pmid)

        print(url)

        urllib.request.urlretrieve(url, out_path)
