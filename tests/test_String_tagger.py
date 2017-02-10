from loctext.learning.annotators import StringTagger
from loctext.learning.train import read_corpus
from loctext.util import PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID


TAGGER_SEND_PARTS = StringTagger(PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, send_whole_once=False)
TAGGER_SEND_WHOLE = StringTagger(PRO_ID, LOC_ID, ORG_ID, UNIPROT_NORM_ID, GO_NORM_ID, TAXONOMY_NORM_ID, send_whole_once=True)

# test when content of documents is sent in parts
def test_annotate_string_tagger_whole_text_false():
    dataset = read_corpus("LocText", corpus_percentage=1.0)
    TAGGER_SEND_PARTS.annotate(dataset)


# test when the whole content of document is sent at once
def test_annotate_string_tagger_whole_text_true():
    dataset = read_corpus("LocText", corpus_percentage=1.0)
    TAGGER_SEND_WHOLE.annotate(dataset)


# get the number of annotated entities in a dataset
def num_predicted_entities(corpus):
    return len(list(corpus.predicted_entities()))


# test if number of annotations of parts is less than or equal to the number of annotations of the whole content
def test_number_of_tagged_entities():
    dataset_parts = read_corpus("LocText", corpus_percentage=1.0)
    dataset_whole = read_corpus("LocText", corpus_percentage=1.0)
    assert 0 == num_predicted_entities(dataset_parts) == num_predicted_entities(dataset_whole)

    TAGGER_SEND_PARTS.annotate(dataset_parts)
    TAGGER_SEND_WHOLE.annotate(dataset_whole)
    # with all organisms explicitly added, parts currently gives one more entity.
    # with only human explicitly added, then the inequality holds
    assert num_predicted_entities(dataset_parts) <= num_predicted_entities(dataset_whole)


# assert statements needs to be written here for verification based on test.
# Many tests are already performed in the String-Tagger.
def test_json_response():
    assert TAGGER_SEND_PARTS.get_string_tagger_json_response("simple text") == {"entities": []}

    assert TAGGER_SEND_PARTS.get_string_tagger_json_response("p53") != {"entities": []}

    assert TAGGER_SEND_PARTS.get_string_tagger_json_response("p53") == {"entities": [
        {"end": 3, "normalizations": [{"id": "FBpp0083753", "type": 7227}, {"id": "", "type": "uniprot:7227"}],
         "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSDARP00000051548", "type": 7955},
                                      {"id": "P79734|P53_DANRE", "type": "uniprot:7955"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSMUSP00000104298", "type": 10090},
                                      {"id": "P02340|P53_MOUSE", "type": "uniprot:10090"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSP00000269305", "type": 9606},
                                      {"id": "P04637|P53_HUMAN", "type": "uniprot:9606"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "FBpp0081732", "type": 7227},
                                      {"id": "O46339|HTH_DROME", "type": "uniprot:7227"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "FBpp0072177", "type": 7227},
                                      {"id": "P08841|TBB3_DROME", "type": "uniprot:7227"}], "start": 1}]}

    assert TAGGER_SEND_WHOLE.get_string_tagger_json_response("p53") == {"entities": [
        {"end": 3, "normalizations": [{"id": "FBpp0083753", "type": 7227}, {"id": "", "type": "uniprot:7227"}],
         "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSDARP00000051548", "type": 7955},
                                      {"id": "P79734|P53_DANRE", "type": "uniprot:7955"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSMUSP00000104298", "type": 10090},
                                      {"id": "P02340|P53_MOUSE", "type": "uniprot:10090"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSP00000269305", "type": 9606},
                                      {"id": "P04637|P53_HUMAN", "type": "uniprot:9606"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "FBpp0081732", "type": 7227},
                                      {"id": "O46339|HTH_DROME", "type": "uniprot:7227"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "FBpp0072177", "type": 7227},
                                      {"id": "P08841|TBB3_DROME", "type": "uniprot:7227"}], "start": 1}]}

    assert TAGGER_SEND_PARTS.get_string_tagger_json_response("p53 nucleus human") == {"entities": [
        {"end": 3, "normalizations": [{"id": "FBpp0083753", "type": 7227}, {"id": "", "type": "uniprot:7227"}],
         "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSDARP00000051548", "type": 7955},
                                      {"id": "P79734|P53_DANRE", "type": "uniprot:7955"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSMUSP00000104298", "type": 10090},
                                      {"id": "P02340|P53_MOUSE", "type": "uniprot:10090"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "ENSP00000269305", "type": 9606},
                                      {"id": "P04637|P53_HUMAN", "type": "uniprot:9606"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "FBpp0081732", "type": 7227},
                                      {"id": "O46339|HTH_DROME", "type": "uniprot:7227"}], "start": 1},
        {"end": 3, "normalizations": [{"id": "FBpp0072177", "type": 7227},
                                      {"id": "P08841|TBB3_DROME", "type": "uniprot:7227"}], "start": 1},
        {"end": 11, "normalizations": [{"id": "GO:0005634", "type": -22}], "start": 5},
        {"end": 17, "normalizations": [{"id": "9606", "type": -3}], "start": 13}]}


# passes if server is running
def test_check_if_is_server_running():
    assert TAGGER_SEND_PARTS.is_server_running("http://localhost:5000/") is True
