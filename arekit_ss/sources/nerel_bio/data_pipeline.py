from collections import OrderedDict

from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.source.nerelbio import labels
from arekit.contrib.source.nerelbio.versions import NerelBioVersions
from arekit.contrib.utils.pipelines.sources.nerel_bio.extrat_text_relations import create_text_relation_extraction_pipeline
from arekit.contrib.utils.pipelines.sources.nerel_bio.labels_fmt import NerelBioAnyLabelFormatter

from arekit_ss.sources.config import SourcesConfig


def build_nerel_bio_datapipeline(cfg):
    """ Data-pipeline is based on the NEREL
    """
    assert(isinstance(cfg, SourcesConfig))

    pipelines, data_folding = create_text_relation_extraction_pipeline(
        nerel_bio_version=NerelBioVersions.V1,
        terms_per_context=cfg.terms_per_context,
        label_formatter=NerelBioAnyLabelFormatter(),
        docs_limit=cfg.docs_limit,
        custom_text_opinion_filters=cfg.optional_filters,
        doc_ops=None,
        text_parser=cfg.text_parser)

    return data_folding, pipelines


class NerelBioAnyLabelScaler(BaseLabelScaler):

    def __init__(self):

        self.__label_to_uint_dict = OrderedDict([
            (labels.ABBREVIATION(), 0),
            (labels.ALTERNATIVE_NAME(), 1),
            (labels.KNOWS(), 2),
            (labels.AGE_IS(), 3),
            (labels.AGE_DIED_AT(), 4),
            (labels.AWARDED_WITH(), 5),
            (labels.PLACE_OF_BIRTH(), 6),
            (labels.DATE_DEFUNCT_IN(), 7),
            (labels.DATE_FOUNDED_IN(), 8),
            (labels.DATE_OF_BIRTH(), 9),
            (labels.DATE_OF_CREATION(), 10),
            (labels.DATE_OF_DEATH(), 11),
            (labels.POINT_IN_TIME(), 12),
            (labels.PLACE_OF_DEATH(), 13),
            (labels.FOUNDED_BY(), 14),
            (labels.HEADQUARTERED_IN(), 15),
            (labels.IDEOLOGY_OF(), 16),
            (labels.SPOUSE(), 17),
            (labels.MEMBER_OF(), 18),
            (labels.ORGANIZES(), 19),
            (labels.OWNER_OF(), 20),
            (labels.PARENT_OF(), 21),
            (labels.PARTICIPANT_IN(), 22),
            (labels.PLACE_RESIDES_IN(), 23),
            (labels.PRICE_OF(), 24),
            (labels.PRODUCES(), 25),
            (labels.RELATIVE(), 26),
            (labels.RELIGION_OF(), 27),
            (labels.SCHOOLS_ATTENDED(), 28),
            (labels.SIBLING(), 29),
            (labels.SUBEVENT_OF(), 30),
            (labels.SUBORDINATE_OF(), 31),
            (labels.TAKES_PLACE_IN(), 32),
            (labels.WORKPLACE(), 33),
            (labels.WORKS_AS(), 34),
            (labels.CONVICTED_OF(), 35),
            (labels.PENALIZED_AS(), 36),
            (labels.START_TIME(), 37),
            (labels.END_TIME(), 38),
            (labels.EXPENDITURE(), 39),
            (labels.AGENT(), 40),
            (labels.INANIMATE_INVOLVED(), 41),
            (labels.INCOME(), 42),
            (labels.SUBCLASS_OF(), 43),
            (labels.PART_OF(), 44),
            (labels.LOCATED_IN(), 45),
            (labels.TREATED_USING(), 46),
            (labels.ORIGINS_FROM(), 47),
            (labels.TO_DETECT_OR_STUDY(), 48),
            (labels.AFFECTS(), 49),
            (labels.HAS_CAUSE(), 50),
            (labels.APPLIED_TO(), 51),
            (labels.USED_IN(), 52),
            (labels.ASSOCIATED_WITH(), 53),
            (labels.HAS_ADMINISTRATION_ROUTE(), 54),
            (labels.HAS_STRENGTH(), 55),
            (labels.DURATION_OF(), 56),
            (labels.VALUE_IS(), 57),
            (labels.PHYSIOLOGY_OF(), 58),
            (labels.PROCEDURE_PERFORMED(), 59),
            (labels.MENTAL_PROCESS_OF(), 60),
            (labels.MEDICAL_CONDITION(), 61),
            (labels.DOSE_IS(), 62),
            (labels.FINDING_OF(), 63),
            (labels.CAUSE_OF_DEATH(), 64),
            (labels.CONSUME(), 65),
        ])

        super(NerelBioAnyLabelScaler, self).__init__(
            uint_dict=self.__label_to_uint_dict,
            int_dict=self.__label_to_uint_dict)
