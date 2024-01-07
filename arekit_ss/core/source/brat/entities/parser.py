from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.partitioning import Partitioning

from arekit_ss.core.source.brat.sentence import BratSentence


class BratTextEntitiesParser(BasePipelineItem):

    def __init__(self, text_fmt="str", **kwargs):
        super(BratTextEntitiesParser, self).__init__(**kwargs)
        self.__partitioning = Partitioning(text_fmt)

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, BratSentence))
        return self.__partitioning.provide(text=input_data.Text, parts_it=input_data.iter_entity_with_local_bounds())
