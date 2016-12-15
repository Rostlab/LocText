#!/usr/bin/env python3
import sys

assert sys.version_info.major == 3, "the script requires Python 3"


__author__ = "Juan Miguel Cejuela (@juanmirocks)"

__help__ = """  Simple parse a GO ontology .obo file.

                This python file can be used as a:
                1) script to filter and extract only a desired GO hierarchy, and
                2) library to parse the parent-children relasionships of the ontology (`is_a` and `part_of`).

                Note: as a script, the GO Terms are printed to standard output. Redirect to a file if needed.

                Note: obsolete terms (`is_obsolete: true`) are kept and parsed for backwards-compatibility.
                Their respective `replaced_by` or `consider`(ed) terms are put in as parents of the obsolete term.
                Importantly note, that some of those replacements are actually not part of the same hierarchy
                of the obsolete go term, example: GO:0016585 is considered by GO:0006338 (a Biological Process).
                Also note, there are some terms that may even have multiple replacements/considerations, e.g GO:0019804.
                Even some obsoleted terms have multiple replacements/considerations, where one is in the hierarchy
                and another one is not.
           """

def parse_arguments(argv=[]):
    import argparse

    parser = argparse.ArgumentParser(description=__help__)

    parser.add_argument('go_file', help='GO ontology file in .obo format, e.g., http://purl.obolibrary.org/obo/go/go-basic.obo')
    parser.add_argument('--hierarchy', required=False, default='', choices=['biological_process', 'molecular_function', 'cellular_component'])

    args = parser.parse_args(argv)

    args.namespace = 'namespace: '+args.hierarchy

    return args


def simple_parse(go_file='', args=None, print_out=False, create_dictionary=True):
    import re

    args = args if args else parse_arguments([go_file])

    print_out = (lambda line: print(line, end='')) if print_out else (lambda _: None)

    dictionary = {} if create_dictionary else None

    state = 'no_term'
    go_id = None
    go_term_is_obsolete = False
    regex_go_id = re.compile('GO:[0-9]+')

    with open(args.go_file) as f:
        for line in f:
            if line.startswith('[Term]'):
                term_token = line
                go_id_line = next(f)
                go_id = regex_go_id.search(go_id_line).group()
                name_line = next(f)
                namespace_line = next(f)

                if namespace_line.startswith(args.namespace):
                    state = 'accept_term'
                    print_out(term_token)
                    print_out(go_id_line)
                    print_out(name_line)
                    print_out(namespace_line)
                else:
                    state = 'ignore_term'

            elif line == f.newlines:
                if state == 'accept_term':
                    if create_dictionary and go_id not in dictionary and not go_term_is_obsolete:
                        # Must be root of respective go hierarchy; no parents
                        dictionary[go_id] = []

                    print_out(line)

                go_term_is_obsolete = False
                state = 'no_term'

            elif state == 'accept_term':
                print_out(line)

                if not go_term_is_obsolete:
                    go_term_is_obsolete = line.startswith("is_obsolete: true")

                if create_dictionary and (
                    line.startswith('is_a: ') or
                        line.startswith('relationship: part_of') or
                        line.startswith('replaced_by:') or
                        line.startswith('consider:')):

                    parent = regex_go_id.search(line).group()
                    parents = dictionary.get(go_id, [])
                    parents = [*parents, parent]
                    dictionary[go_id] = parents

            else:
                continue

    return dictionary


if __name__ == "__main__":
    import sys
    args = parse_arguments(sys.argv[1:])
    simple_parse(args=args, print_out=True, create_dictionary=False)
