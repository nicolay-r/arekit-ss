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
            (labels.ACTIVITY(), 0),
            (labels.MEDPROC(), 0),
            (labels.MONEY(), 0),
            (labels.ADMINISTRATION_ROUTE(), 0),
            (labels.MENTALPROC(), 0),
            (labels.NATIONALITY(), 0),
            (labels.ANATOMY(), 0),
            (labels.PHYS(), 0),
            (labels.NUMBER(), 0),
            (labels.CHEM(), 0),
            (labels.SCIPROC(), 0),
            (labels.ORDINAL(), 0),
            (labels.DEVICE(), 0),
            (labels.AGE(), 0),
            (labels.ORGANIZATION(), 0),
            (labels.DISO(), 0),
            (labels.CITY(), 0),
            (labels.PERCENT(), 0),
            (labels.FINDING(), 0),
            (labels.COUNTRY(), 0),
            (labels.PERSON(), 0),
            (labels.FOOD(), 0),
            (labels.DATE(), 0),
            (labels.PRODUCT(), 0),
            (labels.GENE(), 0),
            (labels.DISTRICT(), 0),
            (labels.PROFESSION(), 0),
            (labels.INJURY_POISONING(), 0),
            (labels.EVENT(), 0),
            (labels.STATE_OR_PROVINCE(), 0),
            (labels.HEALTH_CARE_ACTIVITY(), 0),
            (labels.FAMILY(), 0),
            (labels.TIME(), 0),
            (labels.LABPROC(), 0),
            (labels.FACILITY(), 0),
            (labels.LIVB(), 0),
            (labels.LOCATION(), 0)
        ])

        super(NerelBioAnyLabelScaler, self).__init__(
            uint_dict=self.__label_to_uint_dict,
            int_dict=self.__label_to_uint_dict)
