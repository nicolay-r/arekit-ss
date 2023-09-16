from arekit.common.entities.types import OpinionEntityType
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.text_opinions.base import TextOpinion

from arekit.contrib.utils.entities.filter import EntityFilter
from arekit.contrib.utils.pipelines.text_opinion.filters.base import TextOpinionFilter


class DefaultEntityFilter(EntityFilter):

    def __init__(self, supported_types):
        assert(isinstance(supported_types, list))
        super(DefaultEntityFilter, self).__init__()
        self.__supported_types = supported_types

    def is_ignored(self, entity, e_type):
        return entity.Type not in self.__supported_types


class EntityBasedTextOpinionFilter(TextOpinionFilter):

    def __init__(self, supported_types, is_src=True):
        """ is_src: bool
                indicates whether source or target is expected to be considered.
        """
        super(EntityBasedTextOpinionFilter, self).__init__()
        self.__entity_filter = DefaultEntityFilter(supported_types)
        self.__is_src = is_src

    def filter(self, text_opinion, parsed_doc, entity_service_provider):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(parsed_doc, ParsedDocument))

        if self.__entity_filter is not None:
            text_opinion_end = text_opinion.SourceId if self.__is_src else text_opinion.TargetId
            entity_type = OpinionEntityType.Subject if self.__is_src else OpinionEntityType.Object
            e_source = entity_service_provider._doc_entities[text_opinion_end]
            if self.__entity_filter.is_ignored(e_source, entity_type):
                return False

        return True
