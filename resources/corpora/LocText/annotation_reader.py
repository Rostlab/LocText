import json
import os
import csv

locText_json_files_path = './LocText_anndoc_original_without_normalizations/LocText_master_json/pool'
pubMed_tsv_file_path = './interFile_modified.tsv'
output_json_files_path = './LocText_annjson_with_normalizations'

"""
Finds the annotated file and uses the information in interFile_modified.tsv file and
fills in the normalizations (UniProt, GO, Taxonomy ID) values into the LocText _annjson_ json file.

Example of how normalizations should look like in annjson format:

normalizations: {
    "n_7":{
        "source":{
            "name":"UniProt",
            "id":"P34972",
            "url":null
        },
        "recName":null,
        "confidence":{
            "state":"",
            "who":[
              "ml:dpeker"
            ],
            "prob":0.8458
        }
    }
}
"""

# Fetch each file one by one from "LocText_anndoc_original_without_normalizations" folder
for file_name in os.listdir(locText_json_files_path):
    with open(locText_json_files_path + "/" + file_name) as locText_data_file:
        start = file_name.find("-") + 1
        end = file_name.find(".ann.json")
        pubMed_id = file_name[start:end]
        counter = 0

        # Check if PubMedID is digit or not.
        if pubMed_id.isdigit():
            with open(pubMed_tsv_file_path) as tsv_file:
                tsv_data = csv.reader(tsv_file, delimiter="\t")
                locText_data = json.load(locText_data_file)
                segregated_rows = []

                # Extract only the required rows based on current PubMedID
                for row in tsv_data:
                    if row[0] == pubMed_id:
                        segregated_rows.append(row)

                # For each entity, add corresponding normalization information from segregated row
                for entity in locText_data['entities']:
                    text = entity['offsets'][0]['text']
                    start_index = entity['offsets'][0]['start']

                    for row in segregated_rows:
                        if str(row[1]) == str(text) and str(row[-2]) == str(start_index):
                            obj_type = ""
                            obj_id = ""
                            confidence = entity['confidence']
                            obj = {}

                            if row[2] == "Protein":
                                if row[3] == "Protein":
                                    obj_id = None
                                else:
                                    obj_id = row[3:-3]
                                obj_type = "n_7"
                                name = "UniProt"
                            elif row[2] == "Location":
                                obj_id = row[3:-3]
                                obj_type = "n_8"
                                name = "GO"
                            elif row[2] == "Organism":
                                obj_id = row[3:-3]
                                obj_type = "n_9"
                                name = "Taxonomy"

                            normalization_obj = {
                                obj_type: {"source": {"name": name, "id": obj_id, "url": None}, "recName": None,
                                           "confidence": confidence}}
                            entity['normalizations'] = normalization_obj

                with open(output_json_files_path + "/" + file_name, "w") as output_file:
                    json.dump(locText_data, output_file)

                # break
        else:
            print("File without pubMed file: " + file_name)
