from loctext.util import PRO_ID, LOC_ID, ORG_ID, REL_PRO_LOC_ID, repo_path
from loctext.learning.annotators import LocTextSSmodelRelationExtractor, LocTextDSmodelRelationExtractor, LocTextCombinedModelRelationExtractor
from nalaf.learning.evaluators import DocumentLevelRelationEvaluator, Evaluations
from nalaf import print_verbose, print_debug
from loctext.learning.evaluations import relation_accept_uniprot_go
from nalaf.learning.lib.sklsvm import SklSVM

def parse_arguments(argv=[]):
    import argparse

    parser = argparse.ArgumentParser(description='dooh')

    parser.add_argument('--model', required=True, choices=["SS", "DS", "Combined"])

    parser.add_argument('--corpus', default="LocText", choices=["LocText"])
    parser.add_argument('--corpus_percentage', type=float, required=True, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')
    parser.add_argument('--evaluation_level', type=int, choices=[1, 2, 3, 4], required=True)

    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--k_num_folds', type=int, default=5)

    parser.add_argument('--feature_generators', default='LocText', choices=["LocText", "default"])
    parser.add_argument('--use_tk', default=False, action='store_true')

    parser.add_argument('--minority_class_ss_model', type=int, default=+1, choices=[-1, +1])
    parser.add_argument('--majority_class_undersampling_ss_model', type=float, default=0.9, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    parser.add_argument('--svm_hyperparameter_c_ss_model', action="store", default=0.0080)
    parser.add_argument('--svm_threshold_ss_model', type=float, default=0.0)

    parser.add_argument('--minority_class_ds_model', type=int, default=+1, choices=[-1, +1])
    parser.add_argument('--majority_class_undersampling_ds_model', type=float, default=0.07, help='e.g. 1 == no undersampling; 0.5 == 50% undersampling')
    parser.add_argument('--svm_hyperparameter_c_ds_model', action="store", default=None)
    parser.add_argument('--svm_threshold_ds_model', type=float, default=0.0)

    args = parser.parse_args(argv)

    if args.evaluation_level == 1:
        ENTITY_MAP_FUN = Entity.__repr__
        RELATION_ACCEPT_FUN = str.__eq__
    elif args.evaluation_level == 2:
        ENTITY_MAP_FUN = 'lowercased'
        RELATION_ACCEPT_FUN = str.__eq__
    elif args.evaluation_level == 3:
        ENTITY_MAP_FUN = 'normalized_first'
        RELATION_ACCEPT_FUN = str.__eq__
    elif args.evaluation_level == 4:
        ENTITY_MAP_FUN = 'normalized_first'
        RELATION_ACCEPT_FUN = relation_accept_uniprot_go

    args.evaluator = DocumentLevelRelationEvaluator(rel_type=REL_PRO_LOC_ID, entity_map_fun=ENTITY_MAP_FUN, relation_accept_fun=RELATION_ACCEPT_FUN)

    def set_None_or_typed_argument(argument, expected_type):
        if not argument or argument == 'None':
            return None
        else:
            try:
                return expected_type(argument)
            except Exception as e:
                raise Exception("The argument {} must be of type {}".format(argument, str(expected_type)))

    args.svm_hyperparameter_c_ss_model = set_None_or_typed_argument(args.svm_hyperparameter_c_ss_model, float)
    args.svm_hyperparameter_c_ds_model = set_None_or_typed_argument(args.svm_hyperparameter_c_ds_model, float)

    return args


def parse_arguments_string(arguments=""):
    return parse_arguments(arguments.split("\s+"))


def _select_annotator_model(args):
    # WARN: we should read the class ids from the corpus
    pro_id = PRO_ID
    loc_id = LOC_ID
    rel_id = REL_PRO_LOC_ID

    indirect_feature_generators = {
        "LocText": None,  # Uses annotator's default
        "default": []  # Uses RelationExtractionPipeline's default

    }.get(args.feature_generators)

    # TODO get here: minority_class, majority_class_undersampling, svm_hyperparameter_c = _select_submodel_params(annotator, args)

    ann_switcher = {}

    SS = LocTextSSmodelRelationExtractor(
        pro_id, loc_id, rel_id,
        feature_generators=indirect_feature_generators,
        execute_pipeline=False,
        model=None,
        classification_threshold=args.svm_threshold_ss_model,
        use_tree_kernel=args.use_tk,
        preprocess=True,
        #
        class_weight=None,
        kernel='linear',
        C=1,
    )

    SS.model.set_allowed_feature_keys([0, 1, 4, 5, 9, 11, 12, 13, 14, 15, 17, 18, 19, 22, 24, 26, 27, 30, 31, 32, 34, 38, 39, 40, 41, 45, 46, 49, 50, 51, 54, 56, 62, 65, 66, 71, 73, 77, 79, 80, 86, 87, 89, 90, 97, 98, 101, 104, 105, 108, 110, 112, 115, 117, 119, 120, 121, 122, 125, 126, 132, 136, 137, 141, 142, 143, 145, 148, 149, 151, 152, 153, 155, 157, 159, 160, 161, 162, 163, 164, 166, 168, 169, 170, 171, 172, 178, 182, 183, 185, 193, 196, 199, 208, 210, 211, 212, 213, 219, 225, 229, 230, 232, 233, 235, 236, 237, 238, 240, 241, 242, 244, 246, 248, 249, 254, 260, 261, 264, 265, 268, 276, 277, 283, 284, 286, 290, 294, 295, 298, 299, 301, 305, 311, 313, 319, 321, 326, 337, 338, 339, 340, 341, 343, 344, 345, 349, 354, 369, 370, 371, 377, 385, 386, 388, 389, 391, 392, 393, 395, 396, 405, 406, 407, 409, 410, 411, 412, 413, 415, 417, 418, 430, 432, 433, 434, 435, 440, 441, 444, 445, 446, 450, 451, 453, 460, 462, 463, 464, 465, 467, 468, 470, 471, 472, 473, 474, 475, 477, 479, 480, 482, 483, 484, 485, 486, 487, 497, 502, 504, 505, 506, 507, 508, 514, 515, 516, 517, 518, 519, 523, 527, 528, 532, 534, 544, 552, 553, 554, 555, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 570, 573, 578, 594, 596, 598, 599, 606, 607, 611, 617, 625, 630, 634, 635, 645, 647, 653, 654, 657, 658, 663, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 688, 695, 700, 705, 714, 716, 734, 735, 754, 756, 769, 770, 772, 774, 775, 776, 782, 785, 786, 795, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 833, 840, 842, 843, 844, 845, 854, 858, 859, 861, 867, 874, 885, 888, 889, 891, 899, 908, 917, 920, 923, 935, 938, 971, 976, 977, 978, 979, 980, 981, 982, 996, 1002, 1003, 1006, 1007, 1008, 1014, 1015, 1016, 1024, 1028, 1035, 1036, 1046, 1048, 1051, 1055, 1056, 1057, 1058, 1059, 1063, 1064, 1065, 1066, 1067, 1070, 1071, 1075, 1078, 1079, 1080, 1081, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1100, 1104, 1111, 1125, 1127, 1128, 1131, 1140, 1141, 1143, 1153, 1160, 1166, 1168, 1170, 1171, 1172, 1175, 1178, 1180, 1181, 1182, 1183, 1186, 1187, 1189, 1190, 1191, 1192, 1194, 1195, 1197, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1214, 1215, 1219, 1220, 1221, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1234, 1239, 1241, 1242, 1243, 1244, 1245, 1261, 1262, 1264, 1265, 1266, 1267, 1268, 1273, 1278, 1285, 1288, 1289, 1302, 1303, 1306, 1307, 1308, 1312, 1313, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1326, 1327, 1328, 1330, 1331, 1332, 1334, 1338, 1341, 1342, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1361, 1363, 1366, 1369, 1375, 1376, 1377, 1379, 1380, 1381, 1382, 1383, 1384, 1398, 1399, 1400, 1401, 1402, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1431, 1432, 1436, 1437, 1438, 1464, 1465, 1466, 1467, 1470, 1472, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1485, 1488, 1490, 1492, 1495, 1498, 1499, 1500, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511])
    ann_switcher["SS"] = SS

    # DS = LocTextDSmodelRelationExtractor(pro_id, loc_id, rel_id, feature_generators=indirect_feature_generators, execute_pipeline=False, model=None, classification_threshold=args.svm_threshold_ds_model, use_tree_kernel=args.use_tk)

    if args.model == "Combined":
        ann_switcher["Combined"] = LocTextCombinedModelRelationExtractor(pro_id, loc_id, rel_id, ss_model=ann_switcher["SS"], ds_model=ann_switcher["DS"])

    model = ann_switcher[args.model]
    submodels = _select_annotator_submodels(model)

    return (model, submodels)


def _select_annotator_submodels(model):
    # Simple switch for either single or combined models
    submodels = model.submodels if hasattr(model, 'submodels') else [model]
    return submodels


def _select_submodel_params(annotator, args):

    if isinstance(annotator, LocTextSSmodelRelationExtractor):
        return (args.minority_class_ss_model, args.majority_class_undersampling_ss_model, args.svm_hyperparameter_c_ss_model)

    elif isinstance(annotator, LocTextDSmodelRelationExtractor):
        return (args.minority_class_ds_model, args.majority_class_undersampling_ds_model, args.svm_hyperparameter_c_ds_model)

    raise AssertionError()


def train(training_set, args, annotator_model, submodels, execute_pipeline):

    for index, submodel in enumerate(submodels):
        print("About to train model {}={}".format(index, submodel.__class__.__name__))

        if execute_pipeline:
            submodel.pipeline.execute(training_set, train=True)

        submodel.model.train(training_set)

    return annotator_model.annotate


def evaluate(corpus, args):
    annotator_model, submodels = _select_annotator_model(args)
    is_only_one_model = len(submodels) == 1

    if is_only_one_model:
        annotator_model.pipeline.execute(corpus, train=True)
        annotator_model.model.write_vector_instances(corpus, annotator_model.pipeline.feature_set)

    annotator_gen_fun = (lambda training_set: train(training_set, args, annotator_model, submodels, execute_pipeline=not is_only_one_model))
    evaluator = args.evaluator

    evaluations = Evaluations.cross_validate(annotator_gen_fun, corpus, evaluator, args.k_num_folds, use_validation_set=not args.use_test_set)
    rel_evaluation = evaluations(REL_PRO_LOC_ID).compute(strictness="exact")

    return rel_evaluation


def evaluate_with_argv(argv=[]):
    args = parse_arguments(argv)

    corpus = read_corpus(args.corpus, args.corpus_percentage)

    print_run_args(args, corpus)
    print()
    result = evaluate(corpus, args)
    print()
    print_run_args(args, corpus)
    print_corpus_pipeline_dependent_stats(corpus)
    print()

    return result


def read_corpus(corpus_name, corpus_percentage=1.0):
    import os
    from nalaf.utils.readers import HTMLReader
    from nalaf.utils.annotation_readers import AnnJsonAnnotationReader

    __corpora_dir = repo_path(["resources", "corpora"])

    if corpus_name == "LocText":
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_annjson_with_normalizations/')

    if corpus_name == "LocText_original":
        dir_html = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_plain_html/pool/')
        dir_annjson = os.path.join(__corpora_dir, 'LocText/LocText_anndoc_original_without_normalizations/LocText_master_json/pool/')

    corpus = HTMLReader(dir_html).read()

    if corpus_name.startswith("LocText"):
        # Remove PMCs, full-text
        del corpus.documents["PMC3596250"]
        del corpus.documents["PMC2192646"]
        del corpus.documents["PMC2483532"]
        del corpus.documents["PMC2847216"]

    AnnJsonAnnotationReader(
        dir_annjson,
        read_only_class_id=[PRO_ID, LOC_ID, ORG_ID],
        read_relations=True,
        delete_incomplete_docs=False).annotate(corpus)

    if (corpus_percentage < 1.0):
        corpus, _ = corpus.percentage_split(corpus_percentage)

    return corpus


def print_run_args(args, corpus):
    print("Train Arguments: ")
    for key, value in sorted((vars(args)).items()):
        print("\t{} = {}".format(key, value))

    print_corpus_hard_core_stats(corpus)


def print_corpus_hard_core_stats(corpus):

    print("\nCorpus stats; #docs={} -- #rels={}".format(len(corpus), len(list(corpus.relations()))))


def print_corpus_pipeline_dependent_stats(corpus):

    # Assumes the sentences and edges have been generated (through relations_pipeline)

    T = 0
    P = 0
    N = 0

    for e in corpus.edges():
        T += 1
        if e.is_relation():
            P += 1
        else:
            N += 1

    # Totals for whole corpus (test data too) and with SentenceDistanceEdgeGenerator (only same sentences)
    # abstracts only -- #docs: 100 -- #P=351 vs. #N=308
    # abstract + fulltext -- #docs: 104, P=614 vs N=1480

    print("\t#sentences={}".format(len(list(corpus.sentences()))))
    print("\t#instances: {} : #P={} vs. #N={}".format(T, P, N))
    print("\t#features: {}".format(next(corpus.edges()).features_vector.shape[1]))

    return (P, N)


if __name__ == "__main__":
    import sys
    ret = evaluate_with_argv(sys.argv[1:])
    print(ret)
