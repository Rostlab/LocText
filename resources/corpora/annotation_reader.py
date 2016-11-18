import json
import os

# WARNING the folder names were changes changed after running the script so this are not valid anymores
locText_json_files_path = './LocText/LocText_master_json/pool'
pubMed_json_files_path = './LocText_With_Annotation'
output_json_files_path = './output'

"""
Finds all the annotated file and corresponding LocText _PubAnnotation_ json files and
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

for file_name in os.listdir(locText_json_files_path):
    with open(locText_json_files_path + "/" + file_name) as locText_data_file:
        start = file_name.find("-") + 1
        end = file_name.find(".ann.json")

        if file_name[start:end].isdigit():
            pubMed_file_path = pubMed_json_files_path + "/PubMed-" + str(file_name[start:end]) + ".json"
            if os.path.exists(pubMed_file_path):
                with open(pubMed_file_path) as pubMed_data_file:
                    pubMed_data = json.load(pubMed_data_file)
                    locText_data = json.load(locText_data_file)

                    locText_count = 0

                    if 'denotations' in pubMed_data:
                        pubMed_text = pubMed_data['text']
                        pubMed_lineBreak = pubMed_data['text'].find('\n') + 1

                        for denotation in pubMed_data['denotations']:
                            try:
                                locText_entity = locText_data['entities'][locText_count]
                            except IndexError:
                                break

                            start = denotation['span']['begin']
                            end = denotation['span']['end']
                            protein_or_subCellular_component = pubMed_text[start:end]

                            if start - pubMed_lineBreak > 0:
                                pubMed_offset = start - pubMed_lineBreak
                            else:
                                pubMed_offset = start

                            if locText_entity['offsets'][0]['text'] == protein_or_subCellular_component\
                                    and pubMed_offset == locText_entity['offsets'][0]['start']:
                                obj_type = ""
                                obj_id = ""
                                confidence = locText_entity['confidence']

                                obj = {}

                                if str(denotation['obj']).find('go:GO') != -1:
                                    obj_type1, obj_type, obj_id = str(denotation['obj']).split(":")
                                    obj_type = "n_8"
                                    name = "GO"
                                elif str(denotation['obj']).find('uniprot') != -1 and str(denotation['obj']).find('uniprot:uniprot') == -1:
                                    obj_type, obj_id = str(denotation['obj']).split(":")
                                    obj_type = "n_7"
                                    name = "UniProt"
                                elif str(denotation['obj']).find('GO') != -1:
                                    obj_type, obj_id = str(denotation['obj']).split(":")
                                    obj_type = "n_8"
                                    name = "GO"
                                elif str(denotation['obj']).find('taxonomy') != -1:
                                    obj_type, obj_id = str(denotation['obj']).split(":")
                                    obj_type = "n_9"
                                    name = "Taxonomy"
                                else:
                                    obj_type = "n_7"
                                    name = "UniProt"
                                    obj_id = ""

                                normalization_obj = {obj_type: {"source": {"name": name, "id": obj_id, "url": None}, "recName": None, "confidence": confidence}}
                                locText_entity['normalizations'] = normalization_obj

                            locText_count += 1
                with open(output_json_files_path + "/" + file_name, "w") as output_file:
                    json.dump(locText_data, output_file)
        else:
            print("File without pubMed file: " + file_name)
