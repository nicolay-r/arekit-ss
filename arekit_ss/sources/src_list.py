DATA_PROVIDER_PIPELINES = {
    "ruattitudes": {
        "pipeline": "arekit_ss.sources.ruattitudes.data_pipeline.build_ruattitudes_datapipeline",
        "entity_parser": "arekit.contrib.source.ruattitudes.entity.parser.RuAttitudesTextEntitiesParser",
        "label_scaler": "arekit_ss.sources.labels.scaler.PosNegNeuRelationsLabelScaler",
        "label_formatter": "arekit_ss.sources.labels.formatter.PosNegNeuLabelsFormatter",
        "src_lang": "ru"
    },
    "rusentrel": {
        "pipeline": "arekit_ss.sources.rusentrel.data_pipeline.build_s_rusentrel_datapipeline",
        "entity_parser": "arekit.contrib.source.brat.entities.parser.BratTextEntitiesParser",
        "label_scaler": "arekit_ss.sources.labels.scaler.PosNegNeuRelationsLabelScaler",
        "label_formatter": "arekit_ss.sources.labels.formatter.PosNegNeuLabelsFormatter",
        "src_lang": "ru"
    },
    "sentinerel": {
        "pipeline": "arekit_ss.sources.sentinerel.data_pipeline.build_sentinerel_datapipeline",
        "entity_parser": "arekit.contrib.source.brat.entities.parser.BratTextEntitiesParser",
        "label_scaler": "arekit_ss.sources.labels.scaler.PosNegNeuRelationsLabelScaler",
        "label_formatter": "arekit_ss.sources.labels.formatter.PosNegNeuLabelsFormatter",
        "src_lang": "ru"
    },
    "nerel": {
        "pipeline": "arekit_ss.sources.nerel.data_pipeline.build_nerel_datapipeline",
        "entity_parser": "arekit.contrib.source.brat.entities.parser.BratTextEntitiesParser",
        "label_scaler": "arekit_ss.sources.nerel.data_pipeline.NerelAnyLabelScaler",
        "label_formatter": "arekit.contrib.utils.pipelines.sources.nerel.labels_fmt.NerelAnyLabelFormatter",
        "src_lang": "ru"
    },
    "nerel-bio": {
        "pipeline": "arekit_ss.sources.nerel_bio.data_pipeline.build_nerel_bio_datapipeline",
        "entity_parser": "arekit.contrib.source.brat.entities.parser.BratTextEntitiesParser",
        "label_scaler": "arekit_ss.sources.nerel_bio.data_pipeline.NerelBioAnyLabelScaler",
        "label_formatter": "arekit.contrib.utils.pipelines.sources.nerel_bio.labels_fmt.NerelBioAnyLabelFormatter",
        "src_lang": "ru"
    }
}
