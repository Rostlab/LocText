from loctext.learning.train import read_corpus


def num_normalizations(dataset):
    count = 0
    for data in dataset.entities():
        # Non normalized entities will have "None" value in it. So do not count it.
        if data.normalisation_dict != {'n_7': None}:
            count += 1
    return count


# Test to find number of new normalizations = number of old normalizations + Greens [Newly normalized records by Tanya]
def test_num_of_normalization_in_new_file():

    old_dataset = read_corpus("LocText_v1", corpus_percentage=1.0)
    new_dataset = read_corpus("LocText_v2", corpus_percentage=1.0)

    # 8 is the number of newly normalized records from Tanya.
    assert num_normalizations(new_dataset) == num_normalizations(old_dataset) + 8
    print("Number of newly normalized ID's [Greens] in new file: ", num_normalizations(new_dataset) - num_normalizations(old_dataset))
