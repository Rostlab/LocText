"""
Downloads the necessary corpora for LocText.

Usage: ::

    $ python -m loctext.download_data

"""
if __name__ == '__main__':
    import nltk
    from spacy.en import download

    CORPORA = ['stopwords']

    for corpus in CORPORA:
        nltk.download(corpus)

    download.main(data_size='parser', force=False)
