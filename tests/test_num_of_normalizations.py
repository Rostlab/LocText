from loctext.learning.train import read_corpus

def num_normalizations(dataset):
    count = 0
    for data in dataset.entities():
        if data.normalisation_dict and list(data.normalisation_dict.values())[0]:
            count += 1

    return count


# Test to find number of new normalizations = number of old normalizations + Greens [Newly normalized records by Tanya]
def test_num_of_normalization_in_new_file():

    old_dataset = read_corpus("LocText_v1", corpus_percentage=1.0)
    new_dataset = read_corpus("LocText_v2", corpus_percentage=1.0)

    # First of all, count of entities & relations remains equal (and ofc there are 0 predictions)
    assert(len(list(old_dataset.entities())) == len(list(new_dataset.entities())))
    assert(len(list(old_dataset.predicted_entities())) == len(list(new_dataset.predicted_entities())) == 0)
    assert(len(list(old_dataset.relations())) == len(list(new_dataset.relations())))
    assert(len(list(old_dataset.predicted_relations())) == len(list(new_dataset.predicted_relations())) == 0)

    # Test now new number of normalizations
    print("Actually obtained number of newly normalized ID's: ", num_normalizations(new_dataset) - num_normalizations(old_dataset))

    # 8 is the number of newly normalized records from Tanya, [Greens] in new file.
    assert num_normalizations(new_dataset) == num_normalizations(old_dataset) + 8
