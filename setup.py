from setuptools import setup
from setuptools import find_packages


def readme():
    with open('README.md', encoding='utf-8') as file:
        return file.read()


setup(
    name='LocText',
    version='1.0.0',
    description='NLP-based Relation Extraction (RE) of: Proteins <--> Cell Compartments',
    long_description=readme(),
    url='https://github.com/Rostlab/LocText',
    author='Juan Miguel Cejuela, Shrikant Vinchurkar',
    author_email='loctext@rostlab.org',

    packages=find_packages(exclude=['tests']),

    test_suite='pytest-runner',
    setup_requires=['pytest'],

    dependency_links=[
        'https://github.com/Rostlab/nalaf/tree/develop#egg=nalaf'
    ],

    install_requires=[
        # 'nalaf',
        'scikit-learn == 0.18.1',
        'requests_cache >= 0.4.13',
        'ujson',  # It should be included with spacy, AFAIK
    ]
)
