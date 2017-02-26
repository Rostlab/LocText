def parse(filepath):
    ret = {}
    with open(filepath) as f:
        next(f)  # discard header
        for line in f:
            line = line.strip()
            _, uniprot_ac, go, _, _, _, confirmed, *_ = line.split("\t")
            if not confirmed:
                break
            else:
                rel_key = (uniprot_ac, go)
                ret[rel_key] = confirmed
                print(uniprot_ac, go, confirmed)
    return ret

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    parse(filepath)
