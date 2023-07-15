from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser

import arekit_ss.sources.s_ruattitudes as s_ra
import arekit_ss.sources.s_rusentrel as s_rsr
import arekit_ss.sources.s_sentinerel as s_snr


DATA_PROVIDER_PIPELINES = {
    "ruattitudes": s_ra.build_ruattitudes_datapipeline,
    "rusentrel": s_rsr.build_s_rusentrel_datapipeline,
    "sentinerel": s_snr.build_sentinerel_datapipeline,
}

ENTITY_PARSERS = {
    "ruattitudes": RuAttitudesTextEntitiesParser(),
    "rusentrel": BratTextEntitiesParser(),
    "sentinerel": BratTextEntitiesParser(),
}
