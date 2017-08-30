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
```


## Run


### Sample Script


```shell
python run.py --text "GCN2 was constitutively localized to the nucleolus or recruited to the nucleolus by amino acid starvation stress"
```

You should see something like the following:

```shell
# Predicted entities:
Entity(class_id: e_1, offset: 0, text: GCN2, norms: {'n_7': 'Q9P2K8,Q9LX30,Q9FIB4,P15442'})
Entity(class_id: e_2, offset: 41, text: nucleolus, norms: {'n_9': 'GO:0005730'})
Entity(class_id: e_2, offset: 71, text: nucleolus, norms: {'n_9': 'GO:0005730'})

# Predicted relations:
Relation(class_id:"r_5": e1:"Entity(class_id: e_1, offset: 0, text: GCN2, norms: {'n_7': 'Q9P2K8,Q9LX30,Q9FIB4,P15442'})"   <--->   e2:"Entity(class_id: e_2, offset: 41, text: nucleolus, norms: {'n_9': 'GO:0005730'})")
Relation(class_id:"r_5": e1:"Entity(class_id: e_1, offset: 0, text: GCN2, norms: {'n_7': 'Q9P2K8,Q9LX30,Q9FIB4,P15442'})"   <--->   e2:"Entity(class_id: e_2, offset: 71, text: nucleolus, norms: {'n_9': 'GO:0005730'})")
```

### Python API

Full documentation is due. For now:

* [Read the example script](run.py) to know how to instantiate the main classes
* Traverse over the extracted relations as in [`annotated_corpus.predicted_relations()`](https://github.com/Rostlab/nalaf/blob/develop/nalaf/structures/data.py#L104), which returns objects of [type Relation, as defined in nalaf](https://github.com/Rostlab/nalaf/blob/develop/nalaf/structures/data.py)


For any any issue or question with the [LocText](https://github.com/Rostlab/LocText) and [nalaf](https://github.com/Rostlab/nalaf) code, please open up an issue in the corresponding repository. Indeed, considerable chunks require refactoring and documentation; don't hesitate to complain ;)


## Development

We use [pytest](https://docs.pytest.org/) for testing.

---

To do a quick performance cross-validation of the LocText machine-learning model, execute:

```shell
# Use --help for more possible arguments
python loctext/learning/train.py --model D0
```

In the end, you should see something like:

```shell
Run Arguments:
	corpus_percentage = 1.0
	cv_with_test_set = False
	eval_corpus = None
	evaluate_only_on_edges_plausible_relations = False
	evaluation_level = 4
	evaluator = <nalaf.learning.evaluators.DocumentLevelRelationEvaluator object at 0x10802df98>
	feature_generators = LocText
	force_external_corpus_evaluation = False
	k_num_folds = 5
	load_model = None
	model = D0
	predict_entities = []
	save_model = None
	training_corpus = LocText
	---
	Using libraries versions: numpy == 1.11.2, scipy == 0.18.1, scikit-learn == 0.18.1, spacy == 1.2.0

Training corpus stats:
	#documents: 100
	#relations total: 1345
	#relations prot<-->loc: 550
	#entities: Counter({'e_1': 1393, 'e_2': 558, 'e_3': 277})
	#sentences: 1056
	#instances (edges): 663 -- #P=351 vs. #N=312
	#plausible relations from edges: 351
	#features: 302

# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	214	18	89	0	0	0.9224	0.7063	0.8000	0.0031	0.9224	0.7063	0.8000	0.0031
```
