"""
Downloads the necessary corpora for LocText.

Usage: ::

    $ python -m loctext.download_data

"""
if __name__ == '__main__':
    import nltk

    CORPORA = ['stopwords']

    for corpus in CORPORA:
        nltk.download(corpus)
