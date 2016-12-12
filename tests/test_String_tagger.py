from nalaf.utils.readers import StringReader
from loctext.learning.annotators import StringTagger
import os

def get_string_tagger_post_request_response():

    #content = "Arginine metabolism in Saccharomyces cerevisiae: subcellular localization of the enzymes.\nSubcellular localization of enzymes of arginine metabolism in Sacchar"
    filename = 'stringTaggerData.txt'
    path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/resources/corpora/LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/a.cttjDbKt3vxJIqCWvfhuhVjjfy-15486337.plain.html"
    path_to_file = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/resources/corpora/LocText/'+filename
    with open(path_to_file, 'r') as content_file:
        content = content_file.read()

    dataset = StringReader(content).read()
    #dataset = StringReader(string_tagger_request_text).read()

    # Verify entity offsets - No warnings should be displayed
    dataset.validate_entity_offsets()

    return dataset


def annotate_string_tagger():
    dataset= get_string_tagger_post_request_response()
    #StringTagger().server_is_running()
    StringTagger().annotate(dataset)

#assert statements needs to be written here for verification based on test. Many tests are already performed.
def test_json_response():
    assert StringTagger().get_string_tagger_json_response("simple text") == {"entities":[]}
    assert StringTagger().get_string_tagger_json_response("p53") == {"entities":[{"end": 4,"normalizations":[{"id":"ENSP00000269305","type":9606},{"id":"P04637|P53_HUMAN","type":"uniprot:9606"}],"start":1}]}
    assert StringTagger().get_string_tagger_json_response("p53") != {"entities":[]}

# passes if server is running
def check_if_server_is_running():
    assert StringTagger().server_is_running("http://localhost:5000/")== True
