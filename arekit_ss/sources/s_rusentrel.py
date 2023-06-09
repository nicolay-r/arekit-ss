import itertools

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions, RuSentRelIOUtils
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.utils.pipelines.sources.rusentrel.extract_text_opinions import \
    create_text_opinion_extraction_pipeline

from arekit_ss.sources.config import SourcesConfig
from arekit_ss.sources.labels.sentiment import PositiveTo, NegativeTo


def __iter_doc_ids(version, docs_limit):
    assert((isinstance(docs_limit, int) and docs_limit > 0) or docs_limit is None)
    doc_ids_iter = RuSentRelIOUtils.iter_collection_indices(version)
    if docs_limit is not None:
        doc_ids_iter = itertools.islice(doc_ids_iter, docs_limit)
    return doc_ids_iter


def build_s_rusentrel_datapipeline(cfg):
    assert(isinstance(cfg, SourcesConfig))

    version = RuSentRelVersions.V11

    pipeline = create_text_opinion_extraction_pipeline(
        rusentrel_version=version,
        text_parser=cfg.text_parser,
        labels_fmt=RuSentRelLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))

    data_folding = NoFolding(doc_ids=__iter_doc_ids(version, cfg.docs_limit),
                             supported_data_type=DataType.Train)

    return data_folding, {DataType.Train: pipeline}
