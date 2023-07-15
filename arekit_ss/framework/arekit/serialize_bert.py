from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.experiment.data_type import DataType
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage

from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.bert import BertExperimentInputSerializerPipelineItem


def serialize_bert_pipeline(writer, rows_provider, output_dir):
    assert(isinstance(rows_provider, BaseSampleRowProvider))
    assert(isinstance(output_dir, str))

    return BertExperimentInputSerializerPipelineItem(
        storage=RowCacheStorage(),
        balance_func=lambda _: False,
        samples_io=SamplesIO(target_dir=output_dir, writer=writer),
        save_labels_func=lambda data_type: data_type != DataType.Test,
        rows_provider=rows_provider)