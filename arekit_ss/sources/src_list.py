from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser

import arekit_ss.sources.s_ruattitudes as s_ra
import arekit_ss.sources.s_rusentrel as s_rsr
import arekit_ss.sources.s_sentinerel as s_snr
import arekit_ss.sources.s_nerel as s_sn
from arekit_ss.sources.labels.scaler import PosNegNeuRelationsLabelScaler


DATA_PROVIDER_PIPELINES = {
    "ruattitudes": s_ra.build_ruattitudes_datapipeline,
    "rusentrel": s_rsr.build_s_rusentrel_datapipeline,
    "sentinerel": s_snr.build_sentinerel_datapipeline,
    "nerel": s_sn.build_nerel_datapipeline,
}

ENTITY_PARSERS = {
    "ruattitudes": RuAttitudesTextEntitiesParser(),
    "rusentrel": BratTextEntitiesParser(),
    "sentinerel": BratTextEntitiesParser(),
    "nerel": BratTextEntitiesParser(),
}

LABELS = {
    "ruattitudes": PosNegNeuRelationsLabelScaler(),
    "rusentrel": PosNegNeuRelationsLabelScaler(),
    "sentinerel": PosNegNeuRelationsLabelScaler(),
    "nerel": s_sn.NerelAnyLabelScaler(),
}
