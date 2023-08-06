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
        doc_ops=None,
        text_parser=cfg.text_parser)

    return data_folding, pipelines


class NerelBioAnyLabelScaler(BaseLabelScaler):

    def __init__(self):

        self.__label_to_uint_dict = OrderedDict([
            (labels.ABBREVIATION(), 0),
            (labels.ALTERNATIVE_NAME(), 0),
            (labels.KNOWS(), 0),
            (labels.AGE_IS(), 0),
            (labels.AGE_DIED_AT(), 0),
            (labels.AWARDED_WITH(), 0),
            (labels.PLACE_OF_BIRTH(), 0),
            (labels.DATE_DEFUNCT_IN(), 0),
            (labels.DATE_FOUNDED_IN(), 0),
            (labels.DATE_OF_BIRTH(), 0),
            (labels.DATE_OF_CREATION(), 0),
            (labels.DATE_OF_DEATH(), 0),
            (labels.POINT_IN_TIME(), 0),
            (labels.PLACE_OF_DEATH(), 0),
            (labels.FOUNDED_BY(), 0),
            (labels.HEADQUARTERED_IN(), 0),
            (labels.IDEOLOGY_OF(), 0),
            (labels.SPOUSE(), 0),
            (labels.MEMBER_OF(), 0),
            (labels.ORGANIZES(), 0),
            (labels.OWNER_OF(), 0),
            (labels.PARENT_OF(), 0),
            (labels.PARTICIPANT_IN(), 0),
            (labels.PLACE_RESIDES_IN(), 0),
            (labels.PRICE_OF(), 0),
            (labels.PRODUCES(), 0),
            (labels.RELATIVE(), 0),
            (labels.RELIGION_OF(), 0),
            (labels.SCHOOLS_ATTENDED(), 0),
            (labels.SIBLING(), 0),
            (labels.SUBEVENT_OF(), 0),
            (labels.SUBORDINATE_OF(), 0),
            (labels.TAKES_PLACE_IN(), 0),
            (labels.WORKPLACE(), 0),
            (labels.WORKS_AS(), 0),
            (labels.CONVICTED_OF(), 0),
            (labels.PENALIZED_AS(), 0),
            (labels.START_TIME(), 0),
            (labels.END_TIME(), 0),
            (labels.EXPENDITURE(), 0),
            (labels.AGENT(), 0),
            (labels.INANIMATE_INVOLVED(), 0),
            (labels.INCOME(), 0),
            (labels.SUBCLASS_OF(), 0),
            (labels.PART_OF(), 0),
            (labels.LOCATED_IN(), 0),
            (labels.TREATED_USING(), 0),
            (labels.ORIGINS_FROM(), 0),
            (labels.TO_DETECT_OR_STUDY(), 0),
            (labels.AFFECTS(), 0),
            (labels.HAS_CAUSE(), 0),
            (labels.APPLIED_TO(), 0),
            (labels.USED_IN(), 0),
            (labels.ASSOCIATED_WITH(), 0),
            (labels.HAS_ADMINISTRATION_ROUTE(), 0),
            (labels.HAS_STRENGTH(), 0),
            (labels.DURATION_OF(), 0),
            (labels.VALUE_IS(), 0),
            (labels.PHYSIOLOGY_OF(), 0),
            (labels.PROCEDURE_PERFORMED(), 0),
            (labels.MENTAL_PROCESS_OF(), 0),
            (labels.MEDICAL_CONDITION(), 0),
            (labels.DOSE_IS(), 0),
            (labels.FINDING_OF(), 0),
            (labels.CAUSE_OF_DEATH(), 0),
            (labels.CONSUME(), 0),
        ])

        super(NerelBioAnyLabelScaler, self).__init__(
            uint_dict=self.__label_to_uint_dict,
            int_dict=self.__label_to_uint_dict)
