from loctext.learning.annotators import StringTagger
from loctext.learning.train import read_corpus


# test when content of documents is sent in parts
def test_annotate_string_tagger_whole_text_false():
    dataset = read_corpus("LocText", corpus_percentage=0.04)  # 10116 Rattus norvegicus (rat)
    StringTagger(False, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').annotate(dataset)

 
# test when the whole content of document is sent at once
def test_annotate_string_tagger_whole_text_true():
    dataset = read_corpus("LocText", corpus_percentage=0.04)
    StringTagger(True, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').annotate(dataset)

# get the number of annotated entities in a dataset
def num_annotated_entities(corpus):
    return len(list(corpus.predicted_entities()))


# test if number of annotations of parts is less than or equal to the number of annotations of the whole content
def test_number_of_tagged_entities():
    dataset_false = read_corpus("LocText", corpus_percentage=0.04)
    dataset_true = read_corpus("LocText", corpus_percentage=0.04)
    StringTagger(False, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').annotate(dataset_false)
    StringTagger(True, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').annotate(dataset_true)
    assert num_annotated_entities(dataset_false) <= num_annotated_entities(dataset_true)


# assert statements needs to be written here for verification based on test.
# Many tests are already performed in the String-Tagger.
def test_json_response():
    assert StringTagger(False, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').get_string_tagger_json_response("simple text") == {"entities": []}
    assert StringTagger(False, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').get_string_tagger_json_response("p53") == {"entities": [{"end": 3, "normalizations": [
        {"id": "ENSP00000269305", "type": 9606}, {"id": "P04637|P53_HUMAN", "type": "uniprot:9606"}], "start": 1}]}
    assert StringTagger(False, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').get_string_tagger_json_response("p53") != {"entities": []}
    assert StringTagger(True, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').get_string_tagger_json_response("p53") == {"entities": [{"end": 3, "normalizations": [
        {"id": "ENSP00000269305", "type": 9606}, {"id": "P04637|P53_HUMAN", "type": "uniprot:9606"}], "start": 1}]}
    assert StringTagger(False, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').get_string_tagger_json_response\
               ("p53 nucleus human") == {"entities":[{"end":3,"normalizations":[{"id":"ENSP00000269305","type":9606},
                                                {"id":"P04637|P53_HUMAN","type":"uniprot:9606"}],"start":1},
                                                {"end":11,"normalizations":[{"id":"GO:0005634","type":-22}],"start":5},
                                                {"end":17,"normalizations":[{"id":"9606","type":-3}],"start":13}]}


# passes if server is running
def test_check_if_server_is_running():
    assert StringTagger(False, 'e_1', 'e_2', 'e_3', 'n_7', 'n_8', 'n_9').server_is_running("http://localhost:5000/") is True
