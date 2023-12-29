from arekit.common.docs.parser import DocumentParsers
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.base import SynonymsCollection

from arekit_ss.sources.rusentrel.utils.docs_reader import RuSentRelDocumentsReader
from arekit_ss.sources.rusentrel.utils.io_utils import RuSentRelVersions
from arekit_ss.sources.rusentrel.utils.labels_fmt import RuSentRelLabelsFormatter
from arekit_ss.sources.rusentrel.utils.opinions.collection import RuSentRelOpinions
from labels import PositiveLabel, NegativeLabel


def init_rusentrel_doc(doc_id, pipeline_items, synonyms):
    assert(isinstance(doc_id, int))
    assert(isinstance(synonyms, SynonymsCollection))

    doc = RuSentRelDocumentsReader.read_document(doc_id=doc_id,
                                                 synonyms=synonyms,
                                                 version=RuSentRelVersions.V11)

    parsed_doc = DocumentParsers.parse(doc=doc, pipeline_items=pipeline_items)

    opins_it = RuSentRelOpinions.iter_from_doc(
        doc_id=doc_id,
        labels_fmt=RuSentRelLabelsFormatter(pos_label_type=PositiveLabel,
                                            neg_label_type=NegativeLabel))
    opinions = OpinionCollection(opinions=opins_it,
                                 synonyms=synonyms,
                                 error_on_synonym_end_missed=True,
                                 error_on_duplicates=True)

    return doc, parsed_doc, opinions
