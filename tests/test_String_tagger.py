from nalaf.utils.readers import StringReader
from loctext.learning.annotators import StringTagger
import os

def get_string_tagger_post_request_response():

    #string_tagger_request_text = "Arginine metabolism in Saccharomyces cerevisiae: subcellular localization of the enzymes.\nSubcellular localization of enzymes of arginine metabolism in Sacchar"  #    {"entities":[{"end": 4,"normalizations":[{"id":"ENSP00000269305","type":9606},{"id":"P04637|P53_HUMAN","type":"uniprot:9606"}],"start":1}]}
    filename = 'stringTaggerData.txt'
    with open(os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/resources/corpora/LocText/'+filename, 'r') as content_file:
        content = content_file.read()
    dataset = StringReader(content).read()
    #dataset = StringReader(string_tagger_request_text).read()

    # Verify entity offsets - No warnings should be displayed
    dataset.validate_entity_offsets()

    return dataset

def test_string_tagger():
    dataset= get_string_tagger_post_request_response()
    StringTagger().annotate(dataset)
