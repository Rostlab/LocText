FROM continuumio/anaconda3

RUN apt-get update && \
    apt-get -y install gcc g++ gfortran

RUN git clone https://github.com/Rostlab/nalaf.git && \
    cd nalaf && \
    python setup.py install && \
    python -m nalaf.download_data

RUN cd ..

# We need the following to avoid weird compilation error: https://github.com/explosion/sense2vec/issues/19
RUN conda install libgcc

RUN git clone https://github.com/Rostlab/LocText.git && \
    cd LocText && \
    python setup.py develop && \
    python -m loctext.download_data && \
    python -m spacy download en
