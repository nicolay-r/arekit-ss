from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.utils.pipelines.sources.nerel.labels_fmt import NerelAnyLabelFormatter
from arekit.contrib.utils.pipelines.sources.nerel_bio.labels_fmt import NerelBioAnyLabelFormatter

from arekit_ss.sources.labels.formatter import PosNegNeuLabelsFormatter
from arekit_ss.sources.labels.scaler import PosNegNeuRelationsLabelScaler
from arekit_ss.sources.nerel.data_pipeline import build_nerel_datapipeline, NerelAnyLabelScaler
from arekit_ss.sources.nerel_bio.data_pipeline import build_nerel_bio_datapipeline, NerelBioAnyLabelScaler
from arekit_ss.sources.ruattitudes.data_pipeline import build_ruattitudes_datapipeline
from arekit_ss.sources.rusentrel.data_pipeline import build_s_rusentrel_datapipeline
from arekit_ss.sources.sentinerel.data_pipeline import build_sentinerel_datapipeline


DATA_PROVIDER_PIPELINES = {
    "ruattitudes": {
        "pipeline": build_ruattitudes_datapipeline,
        "entity_parser": RuAttitudesTextEntitiesParser(),
        "label_scaler": PosNegNeuRelationsLabelScaler(),
        "label_formatter": PosNegNeuLabelsFormatter(),
        "src_lang": "ru"
    },
    "rusentrel": {
        "pipeline": build_s_rusentrel_datapipeline,
        "entity_parser": BratTextEntitiesParser(),
        "label_scaler": PosNegNeuRelationsLabelScaler(),
        "label_formatter": PosNegNeuLabelsFormatter(),
        "src_lang": "ru"
    },
    "sentinerel": {
        "pipeline": build_sentinerel_datapipeline,
        "entity_parser": BratTextEntitiesParser(),
        "label_scaler": PosNegNeuRelationsLabelScaler(),
        "label_formatter": PosNegNeuLabelsFormatter(),
        "src_lang": "ru"
    },
    "nerel": {
        "pipeline": build_nerel_datapipeline,
        "entity_parser": BratTextEntitiesParser(),
        "label_scaler": NerelAnyLabelScaler(),
        "label_formatter": NerelAnyLabelFormatter(),
        "src_lang": "ru"
    },
    "nerel-bio": {
        "pipeline": build_nerel_bio_datapipeline,
        "entity_parser": BratTextEntitiesParser(),
        "label_scaler": NerelBioAnyLabelScaler(),
        "label_formatter": NerelBioAnyLabelFormatter(),
        "src_lang": "ru"
    }
}
