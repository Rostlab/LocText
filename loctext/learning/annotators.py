from nalaf.learning.taggers import RelationExtractor
from nalaf.structures.dataset_pipelines import PrepareDatasetPipeline
from nalaf.learning.taggers import StubSameSentenceRelationExtractor

class LocTextBaselineRelationExtractor(RelationExtractor):

    def __init__(
        self,
        entity1_class,
        entity2_class,
        rel_type):

        super().__init__(entity1_class, entity2_class, rel_type)
        self.__annotator = StubSameSentenceRelationExtractor(entity1_class, entity2_class, rel_type)

    def annotate(self, corpus):
        return self.__annotator.tag(corpus)  # TODO change to annotate


class LocTextRelationExtractor(RelationExtractor):

    @staticmethod
    def default_features_pipeline():
        return PrepareDatasetPipeline()

    def __init__(
            self,
            entity1_class,
            entity2_class,
            rel_type,
            bin_model,
            svmlight=None,
            pipeline=None,
            execute_pipeline=True):

        super().__init__(entity1_class, entity2_class, rel_type)
        self.bin_model = bin_model
        self.pipeline = pipeline if pipeline else LocTextRelationExtractor.default_features_pipeline
        self.execute_pipeline = execute_pipeline
        # ---
        SVM_PATH = '/usr/local/manual/bin/'  # TODO hardcoded
        self.svmlight = svmlight if svmlight else SVMLightTreeKernels(SVM_PATH, self.bin_model, use_tree_kernel=False)


    def annotate(self, corpus):
        if self.execute_pipeline:
            # self.pipeline.execute(corpus, train=False, feature_set=feature_set)
            self.pipeline.execute(dataset)

        feature_set = -1  # undefined yet

        svmlight.create_input_file(corpus, 'predict', feature_set)
        svmlight.tag(mode='predict')
        svmlight.read_predictions(corpus)

        return corpus
