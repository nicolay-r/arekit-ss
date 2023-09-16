import itertools

from arekit.common.experiment.data_type import DataType
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions, RuSentRelIOUtils
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.utils.pipelines.sources.rusentrel.extract_text_opinions import \
    create_text_opinion_extraction_pipeline

from arekit_ss.sources.config import SourcesConfig
from arekit_ss.sources.labels.sentiment import PositiveTo, NegativeTo


def __iter_doc_ids(doc_ids_iter, docs_limit):
    assert((isinstance(docs_limit, int) and docs_limit > 0) or docs_limit is None)
    if docs_limit is not None:
        doc_ids_iter = itertools.islice(doc_ids_iter, docs_limit)
    return doc_ids_iter


def build_s_rusentrel_datapipeline(cfg):
    assert(isinstance(cfg, SourcesConfig))

    version = RuSentRelVersions.V11

    pipeline = create_text_opinion_extraction_pipeline(
        rusentrel_version=version,
        text_parser=cfg.text_parser,
        custom_text_opinion_filters=cfg.optional_filters,
        labels_fmt=RuSentRelLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))

    data_folding = {
        DataType.Train: list(__iter_doc_ids(RuSentRelIOUtils.iter_train_indices(version), cfg.docs_limit)),
        DataType.Test: list(__iter_doc_ids(RuSentRelIOUtils.iter_test_indices(version), cfg.docs_limit))
    }

    data_pipeline = {
        DataType.Train: pipeline,
        DataType.Test: pipeline
    }

    return data_folding, data_pipeline
