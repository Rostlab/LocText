import csv

OLD_TSV_FILE_NAME = "../resources/corpora/LocText/interFile_modified.tsv"
NEW_TSV_FILE_NAME = "../resources/corpora/LocText/interFile_modified_by_Tanya.tsv"


# Count number of ID's which have normalizations
def count_normalized_id(filename):
    alist = []
    with open(filename) as tsv_file:
        # Skip first line
        data = dict(enumerate(csv.reader(tsv_file, delimiter='\t')))
        for row in data:
            if data[row][3] != "Protein":
                alist.append(row)
        return len(alist)


# Count number of ID's which does not have normalizations
def count_non_normalized_id(filename):
    alist = []
    with open(filename) as tsv_file:
        # Skip first line
        data = dict(enumerate(csv.reader(tsv_file, delimiter='\t')))
        for row in data:
            if data[row][3] == 'Protein':
                alist.append(row)
        return len(alist)


# Count number of newly created records.
def count_newly_normalized_records():
    return count_non_normalized_id(OLD_TSV_FILE_NAME) - count_non_normalized_id(NEW_TSV_FILE_NAME)


# Test to find number of new normalizations = new of old normalizations + Greens [Newly normalized records]
def test_num_of_normalization_in_new_file():
    num_of_normalization_in_new_file = count_normalized_id(NEW_TSV_FILE_NAME)
    num_of_normalization_in_old_file = count_normalized_id(OLD_TSV_FILE_NAME)
    assert (num_of_normalization_in_new_file == num_of_normalization_in_old_file + count_newly_normalized_records()), \
            "the number of normalizations in the new corpus is not equal to number of normalizations of " \
            "the previous corpus + the number of greens [Newly normalized id count]"
    print("Number of newly normalized ID's [Greens] in new file: ", count_newly_normalized_records())
