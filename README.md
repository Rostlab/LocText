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


## Run (examples)
