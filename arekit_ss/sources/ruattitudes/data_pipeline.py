from arekit.common.experiment.data_type import DataType

from arekit_ss.sources.config import SourcesConfig
from arekit_ss.sources.labels.scaler import PosNegNeuRelationsLabelScaler
from arekit_ss.sources.ruattitudes.doc_provider import RuAttitudesDocumentProvider
from arekit_ss.sources.ruattitudes.extract_text_opinions import create_text_opinion_extraction_pipeline
from arekit_ss.sources.ruattitudes.utils.io_utils import RuAttitudesVersions


def build_ruattitudes_datapipeline(cfg):
    assert(isinstance(cfg, SourcesConfig))

    version = RuAttitudesVersions.V20Large

    pipeline = create_text_opinion_extraction_pipeline(
        version=version,
        text_parser=cfg.text_parser_items,
        label_scaler=PosNegNeuRelationsLabelScaler(),
        custom_text_opinion_filters=cfg.optional_filters,
        limit=cfg.docs_limit)

    d = RuAttitudesDocumentProvider.read_ruattitudes_to_brat_in_memory(
        version=version,
        keep_doc_ids_only=True,
        doc_id_func=lambda doc_id: doc_id,
        limit=cfg.docs_limit)

    data_folding = {
        DataType.Train: list(d.keys())
    }

    return data_folding, {DataType.Train: pipeline}
