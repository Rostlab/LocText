from nalaf.learning.taggers import RelationExtractor
from nalaf.structures.dataset_pipelines import PrepareDatasetPipeline
from nalaf.learning.taggers import StubSameSentenceRelationExtractor
from nalaf.learning.svmlight import SVMLightTreeKernels
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.features.relations import NamedEntityCountFeatureGenerator


class LocTextBaselineRelationExtractor(RelationExtractor):

    def __init__(
        self,
        entity1_class,
        entity2_class,
        rel_type):

        super().__init__(entity1_class, entity2_class, rel_type)
        self.__annotator = StubSameSentenceRelationExtractor(entity1_class, entity2_class, rel_type)

    def annotate(self, corpus):
        return self.__annotator.annotate(corpus)


class LocTextRelationExtractor(RelationExtractor):

    @staticmethod
    def default_feature_generators(class1, class2, graphs=None):

        #GRAPHS_CLOSURE_VARIABLE = {} if graphs is None else graphs

        return [
            NamedEntityCountFeatureGenerator(class1),
            NamedEntityCountFeatureGenerator(class2)
        ]


    def __init__(
            self,
            entity1_class,
            entity2_class,
            rel_type,
            bin_model,
            pipeline=None,
            svmlight=None,
            execute_pipeline=True):

        super().__init__(entity1_class, entity2_class, rel_type)
        self.bin_model = bin_model
        self.svmlight = svmlight if svmlight else SVMLightTreeKernels(model_path=self.bin_model, use_tree_kernel=False)
        self.pipeline = pipeline if pipeline else RelationExtractionPipeline(entity1_class, entity2_class, rel_type, feature_generators=LocTextRelationExtractor.default_feature_generators(self.entity1_class, self.entity2_class))
        self.execute_pipeline = execute_pipeline


    def annotate(self, corpus):
        if self.execute_pipeline:
            self.pipeline.execute(corpus, train=False)

        instancesfile = self.svmlight.create_input_file(corpus, 'predict', self.pipeline.feature_set)
        predictionsfile = self.svmlight.tag(instancesfile)
        self.svmlight.read_predictions(corpus, predictionsfile)

        return corpus
