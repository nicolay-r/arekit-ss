from arekit.common.data.const import ENTITIES, ENTITY_TYPES
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.pipelines.items.sampling.base import BaseSerializerPipelineItem


def serialize_bert_pipeline(rows_provider, samples_io):
    assert(isinstance(rows_provider, BaseSampleRowProvider))

    return BaseSerializerPipelineItem(
        storage=RowCacheStorage(force_collect_columns=[ENTITIES, ENTITY_TYPES]),
        samples_io=samples_io,
        save_labels_func=lambda _: True,
        rows_provider=rows_provider,
        src_key=None)
