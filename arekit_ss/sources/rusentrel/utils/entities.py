from arekit.common.entities.collection import EntityCollection
from arekit.common.synonyms.base import SynonymsCollection

from arekit_ss.core.source.brat.annot import BratAnnotationParser
from arekit_ss.sources.rusentrel.utils.io_utils import RuSentRelIOUtils, RuSentRelVersions


class RuSentRelDocumentEntityCollection(EntityCollection):

    def __init__(self, entities, value_to_group_id_func):
        super(RuSentRelDocumentEntityCollection, self).__init__(
            entities=entities,
            value_to_group_id_func=value_to_group_id_func)

        self._sort_entities(key=lambda entity: entity.IndexBegin)

    @classmethod
    def read_collection(cls, doc_id, synonyms, version=RuSentRelVersions.V11):
        assert (isinstance(synonyms, SynonymsCollection))
        assert (isinstance(doc_id, int))

        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_entity_innerpath(index=doc_id, version=version),
            process_func=lambda input_file: cls(
                entities=BratAnnotationParser.parse_annotations(input_file)["entities"],
                value_to_group_id_func=synonyms.get_synonym_group_index),
            version=version)
