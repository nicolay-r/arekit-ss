from collections import OrderedDict

from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.source.nerel import labels
from arekit.contrib.source.nerel.io_utils import NerelVersions
from arekit.contrib.utils.pipelines.sources.nerel.extract_text_relations import create_text_relation_extraction_pipeline

from arekit_ss.sources.config import SourcesConfig


def build_nerel_datapipeline(cfg):
    assert(isinstance(cfg, SourcesConfig))

    pipelines, data_folding = create_text_relation_extraction_pipeline(
        nerel_version=NerelVersions.V11,
        terms_per_context=cfg.terms_per_context,
        docs_limit=cfg.docs_limit,
        doc_ops=None,
        text_parser=cfg.text_parser)

    return data_folding, pipelines


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
            (labels.STATE_BELONGS_TO(), 8),
            (labels.PosAuthorFrom(), 9),
            (labels.NegAuthorFrom(), 10),
            (labels.ALTERNATIVE_NAME(), 11),
            (labels.ORIGINS_FROM(), 12),
            (labels.START_TIME(), 13),
            (labels.OWNER_OF(), 14),
            (labels.SUBEVENT_OF(), 15),
            (labels.PARENT_OF(), 16),
            (labels.SUBORDINATE_OF(), 17),
            (labels.PART_OF(), 18),
            (labels.TAKES_PLACE_IN(), 19),
            (labels.PARTICIPANT_IN(), 20),
            (labels.WORKPLACE(), 21),
            (labels.PENALIZED_AS(), 22),
            (labels.WORKS_AS(), 23),
            (labels.PLACE_OF_DEATH(), 24),
            (labels.PLACE_OF_BIRTH(), 25),
            (labels.HAS_CAUSE(), 26),
            (labels.AWARDED_WITH(), 27),
            (labels.CAUSE_OF_DEATH(), 28),
            (labels.CONVICTED_OF(), 29),
            (labels.DATE_DEFUNCT_IN(), 30),
            (labels.DATE_FOUNDED_IN(), 31),
            (labels.DATE_OF_BIRTH(), 32),
            (labels.DATE_OF_CREATION(), 33),
            (labels.DATE_OF_DEATH(), 34),
            (labels.END_TIME(), 35),
            (labels.EXPENDITURE(), 36),
            (labels.FOUNDED_BY(), 37),
            (labels.KNOWS(), 38),
            (labels.RELATIVE(), 39),
            (labels.LOCATED_IN(), 40),
            (labels.RELIGION_OF(), 41),
            (labels.MEDICAL_CONDITION(), 42),
            (labels.SCHOOLS_ATTENDED(), 43),
            (labels.MEMBER_OF(), 44),
            (labels.SIBLING(), 45),
            (labels.ORGANIZES(), 46),
            (labels.SPOUSE(), 47)
        ])

        super(NerelAnyLabelScaler, self).__init__(
            uint_dict=self.__uint_to_label_dict,
            int_dict=self.__uint_to_label_dict)
