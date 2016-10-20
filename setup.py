from setuptools import setup
from setuptools import find_packages


def readme():
    with open('README.md', encoding='utf-8') as file:
        return file.read()

setup(
    name='nala',
    version='0.1.0',
    description='NLP Extraction of Relationships: Protein--Cell Compartments',
    long_description=readme(),
    url='https://github.com/Rostlab/LocText',
    author='Shrikant Vinchurkar, Juan Miguel Cejuela, Ashish Baghudana',
    author_email='loctext@rostlab.org',
    packages=find_packages(exclude=['tests']),
    test_suite='nose.collector',
    setup_requires=['nose>=1.0'],

    dependency_links=[
        'https://github.com/Rostlab/nalaf/tree/develop#egg=nalaf'
    ],

    install_requires=[
        'nalaf',
        'spacy'
    ]
)
