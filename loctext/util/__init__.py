import os

PRO_ID = 'e_1'
LOC_ID = 'e_2'
ORG_ID = 'e_3'
REL_PRO_LOC_ID = 'r_5'

UNIPROT_NORM_ID = 'n_7'
GO_NORM_ID = 'n_9'
TAXONOMY_NORM_ID = 'n_8'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def repo_path(*args):
    return os.path.join(REPO_ROOT, *args)


def reversed_feature_set_map(feature_set):
    """
    Return the reversed map of the feature set (as in a bidirectional map -- bijection).

    That is, the original feature_set mapped (feature name) --> (feature key).
    This function, returns the dictionary that maps (feature key) --> (feature map).

    """
    return dict(map(reversed, feature_set.items()))


def print_selected_features(selected_feat_keys, feature_set, file_prefix, file_date=None):
    import time

    if file_date is None:
        file_date = str(time.time())

    all_filenames_prefix = "{}-{}-".format(file_prefix, file_date)

    reversed_feature_set = reversed_feature_set_map(feature_set)
    selected_feat_names = [reversed_feature_set[feat_key] for feat_key in selected_feat_keys]

    def _pickle_beautified_file(name, feats):
        file_name = all_filenames_prefix + name + ".log"
        with open(file_name, 'w') as f:
            def _print(string=""):
                print(string, file=f)
            def _print_feat(feat, index):
                if name == "KEYS":  # --> int ; without quotes
                    format_str = '    {},  # {}'.format(feat, index)
                else:  # NAMES --> str ; within quotes
                    format_str = '    "{}",  # {}'.format(feat, index)
                _print(format_str)

            _print("[")

            for index, feat in enumerate(feats):
                _print_feat(feat, index)

            _print("]")

        return file_name

    names_file = _pickle_beautified_file("NAMES", selected_feat_names)
    fig_file = all_filenames_prefix + "FIGURE.pdf"

    return (names_file, fig_file)


def unpickle_beautified_file(file_path, k_best=None):
    import ast
    from pathlib import Path
    ret = ast.literal_eval(Path(file_path).read_text())
    if k_best is not None:
        return ret[:k_best]
    else:
        return ret
