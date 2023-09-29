from arekit.common.data.const import ENTITIES, ENTITY_TYPES
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage

from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.base import BaseSerializerPipelineItem


def serialize_bert_pipeline(writer, rows_provider, output_dir):
    assert(isinstance(rows_provider, BaseSampleRowProvider))
    assert(isinstance(output_dir, str))

    return BaseSerializerPipelineItem(
        storage=RowCacheStorage(force_collect_columns=[ENTITIES, ENTITY_TYPES]),
        samples_io=SamplesIO(target_dir=output_dir, writer=writer),
        save_labels_func=lambda _: True,
        rows_provider=rows_provider)
