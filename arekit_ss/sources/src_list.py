from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.utils.pipelines.sources.nerel.labels_fmt import NerelAnyLabelFormatter

from arekit_ss.sources.labels.formatter import PosNegNeuLabelsFormatter
from arekit_ss.sources.labels.scaler import PosNegNeuRelationsLabelScaler
from arekit_ss.sources.nerel.data_pipeline import build_nerel_datapipeline, NerelAnyLabelScaler
from arekit_ss.sources.ruattitudes.data_pipeline import build_ruattitudes_datapipeline
from arekit_ss.sources.rusentrel.data_pipeline import build_s_rusentrel_datapipeline
from arekit_ss.sources.sentinerel.data_pipeline import build_sentinerel_datapipeline


DATA_PROVIDER_PIPELINES = {
    "ruattitudes": {
        "pipeline": build_ruattitudes_datapipeline,
        "entity_parser": RuAttitudesTextEntitiesParser(),
        "label_scaler": PosNegNeuRelationsLabelScaler(),
        "label_formatter": PosNegNeuLabelsFormatter()
    },
    "rusentrel": {
        "pipeline": build_s_rusentrel_datapipeline,
        "entity_parser": BratTextEntitiesParser(),
        "label_scaler": PosNegNeuRelationsLabelScaler(),
        "label_formatter": PosNegNeuLabelsFormatter(),
    },
    "sentinerel": {
        "pipeline": build_sentinerel_datapipeline,
        "entity_parser": BratTextEntitiesParser(),
        "label_scaler": PosNegNeuRelationsLabelScaler(),
        "label_formatter": PosNegNeuLabelsFormatter(),
    },
    "nerel": {
        "pipeline": build_nerel_datapipeline,
        "entity_parser": BratTextEntitiesParser(),
        "label_scaler": NerelAnyLabelScaler(),
        "label_formatter": NerelAnyLabelFormatter()
    }
}
