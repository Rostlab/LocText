import pyexcel
import csv
from collections import defaultdict, OrderedDict

'''
Helper functions for dealing with xlsx and tsv files. The operations below are supported:
1. Conversion of xlsx to tsv file
2. Creation of new tsv file that contains only normalized rows
3. Finding of differences between two tsv files and occurrences of non-normalized rows in those files

IMPORTANT:
For xlsx conversion, you need to install the following:
pip install pyexcel
pip install pyexcel-xlsx

'''

# -------------- Convert a xlsx file to a tsv file -----------------------------------------------------

xlsx_file = "file_name.xlsx"
tsv_file = 'new_file_name.tsv'


def convert_file(xlsx_file, tsv_file):
    data = pyexcel.get_array(file_name=xlsx_file)
    pyexcel.save_as(array=data,
                    dest_file_name=tsv_file,
                    dest_delimiter='\t')


convert_file(xlsx_file, tsv_file)


# --------------- Create a new file that contains only normalized rows ----------------------------------

# create a list of not-normalized rows
def not_normalized_rows(input_file):
    a_list = []
    with open(input_file) as tsv_file:
        data = dict(enumerate(csv.reader(tsv_file, delimiter='\t')))
        for row in data:
            if data[row][3] != 'Protein' and data[row][2] == 'Protein':
                a_list.append(row)
    return a_list  # len(a_list)) if number of not-normalized rows is needed


file = 'interFile_modified_by_Tanya.tsv'
new_file = 'new_file.tsv'

first_row = 1
rows_to_delete = not_normalized_rows(file)


# create the new file
def create_normalized_file(in_file, out_file):
    with open(in_file, 'rt') as infile, open(out_file, 'wt') as out:
        out.writelines(row for row_num, row in enumerate(infile, first_row)
                       if row_num not in rows_to_delete)


create_normalized_file(file, new_file)


# -------------- Find differences between two tsv files and occurrences of non-normalized rows in those files ----------
def find_occurrences(input_file):
    a_set = set()

    with open(input_file) as tsv_file:
        data = dict(enumerate(csv.reader(tsv_file, delimiter='\t')))
        for row in data:
            a_set.add(data[row][0])  # get all PubmedIDs

    pubmed_ID = list(a_set)
    count_ID = list()

    with open(input_file) as tsv_file:
        data = dict(enumerate(csv.reader(tsv_file, delimiter='\t')))
        for i in range(101):
            for row in data:
                if data[row][0] == pubmed_ID[i] and data[row][3] == 'Protein':
                    count_ID.append(data[row][0])  # get number of rows for non-normalized pubmedIDs

    not_normalized = defaultdict(int)
    for curr in count_ID:
        not_normalized[curr] += 1

    return not_normalized


# files
file1 = 'interFile_modified.tsv'
file2 = 'interFile_modified_by_Tanya.tsv'

# check the differences
print(OrderedDict(find_occurrences(file1)))
print(OrderedDict(find_occurrences(file2)))


# FILE1 = './interFile_modified_by_Tanya.tsv'
# FILE2 ='./interFile_modified.tsv'
#
# # create a list of not-normalized rows
# def not_normalized_rows(input_file):
#     a_list = []
#     with open(input_file) as tsv_file:
#         data = dict(enumerate(csv.reader(tsv_file, delimiter='\t')))
#         for row in data:
#             if data[row][2] == 'Location':
#             # if data[row][3] == 'Protein' and data[row][2] == 'Protein':
#                 a_list.append(row)
#     return a_list  # len(a_list)) if number of not-normalized rows is needed
#
# print(len(not_normalized_rows(FILE1)))
# print(len(not_normalized_rows(FILE2)))