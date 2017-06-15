[![Build Status](https://travis-ci.org/Rostlab/LocText.svg?branch=develop)](https://travis-ci.org/Rostlab/LocText)
[![codecov](https://codecov.io/gh/Rostlab/LocText/branch/develop/graph/badge.svg)](https://codecov.io/gh/Rostlab/LocText)

# LocText

Text-mine the relationship of `Proteins <--> Cell Compartments` (meaning, _protein is/functions in cell compartment_).

Run on PubMed abstracts or any string (i.e. including full text).

## Requirements

Runs on `Python >= 3.5`.

Non-packaged dependencies (each software has its own dependencies):

* [nalaf](https://github.com/Rostlab/nalaf)
* [STRING tagger](https://github.com/juanmirocks/STRING-tagger-server)


## Install

```shell
git clone https://github.com/Rostlab/LocText.git
cd LocText
pip3 install .
python -m loctext.download_data
python -m spacy download en
```


## Run


### Sample Script


```shell
python run.py --text "GCN2 was constitutively localized to the nucleolus or recruited to the nucleolus by amino acid starvation stress"

python run.py --pmid 28189689  # Annotate the abstract of the given PMID -- Note, sometimes this may fail if the the online NCBI eutils are down
```


### Python API

Full documentation is due. For now:

* [Read the example script](run.py) to know how to instantiate the main classes
* Traverse over the extracted relations as in [`annotated_corpus.predicted_relations()`](https://github.com/Rostlab/nalaf/blob/develop/nalaf/structures/data.py#L104), which returns objects of [type Relation, as defined in nalaf](https://github.com/Rostlab/nalaf/blob/develop/nalaf/structures/data.py#L1963)


For any any issue or question with the LocText and nalaf code, please open up an issue in the corresponding repository. Indeed, considerable chunks require refactoring and documentation; don't hesitate to complain ;)
