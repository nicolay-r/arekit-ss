from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.utils.pipelines.text_opinion.filters.base import TextOpinionFilter


class LabelTextOpinionFilter(TextOpinionFilter):

    def __init__(self, relation_types):
        assert(isinstance(relation_types, list))
        self.__relation_types = set(relation_types)

    def filter(self, text_opinion, parsed_doc, entity_service_provider):
        assert(isinstance(text_opinion, TextOpinion))
        return type(text_opinion.Label).__name__ in self.__relation_types
