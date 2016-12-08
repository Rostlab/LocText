from nalaf.utils.readers import StringReader
from loctext.learning.annotators import StringTagger

def get_string_tagger_post_request_response():

    string_tagger_request_text = ""

    dataset = StringReader(string_tagger_request_text).read()

    # Verify entity offsets - No warnings should be displayed
    dataset.validate_entity_offsets()

    return dataset

def test_string_tagger():
    dataset = get_string_tagger_post_request_response()

    StringTagger().annotate(dataset)

    # assert statements needs to be written here for verification based on test.



