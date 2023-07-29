from collections import OrderedDict

from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.source.nerel import labels
from arekit.contrib.source.nerel.io_utils import NerelVersions
from arekit.contrib.utils.pipelines.sources.nerel.extract_text_relations import create_text_relation_extraction_pipeline

from arekit_ss.sources.config import SourcesConfig


def build_nerel_datapipeline(cfg):
    assert(isinstance(cfg, SourcesConfig))

    pipelines, data_folding = create_text_relation_extraction_pipeline(
        sentinerel_version=NerelVersions.V11,
        terms_per_context=cfg.terms_per_context,
        docs_limit=cfg.docs_limit,
        doc_ops=None,
        text_parser=cfg.text_parser)

    return data_folding, {DataType.Train: pipelines[DataType.Train]}


class NerelAnyLabelScaler(BaseLabelScaler):

    def __init__(self):

        self.__uint_to_label_dict = OrderedDict([
            (labels.OpinionBelongsTo(), 0),
            (labels.OpinionRelatesTo(), 1),
            (labels.NegEffectFrom(), 2),
            (labels.PosEffectFrom(), 3),
            (labels.NegStateFrom(), 4),
            (labels.PosStateFrom(), 5),
            (labels.NegativeTo(), 6),
            (labels.PositiveTo(), 7),
            (labels.StateBelongsTo(), 8),
            (labels.PosAuthorFrom(), 9),
            (labels.NegAuthorFrom(), 10),
            (labels.AlternativeName(), 11),
            (labels.OriginsFrom(), 12),
        ])

        super(NerelAnyLabelScaler, self).__init__(
            uint_dict=self.__uint_to_label_dict,
            int_dict=self.__uint_to_label_dict)
