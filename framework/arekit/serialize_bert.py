from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage

from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.bert import BertExperimentInputSerializerPipelineItem


def serialize_bert(writer, sample_row_provider, output_dir, data_folding, data_type_pipelines):
    assert(isinstance(sample_row_provider, BaseSampleRowProvider))
    assert(isinstance(output_dir, str))

    pipeline = BasePipeline([
        BertExperimentInputSerializerPipelineItem(
            storage=RowCacheStorage(),
            balance_func=lambda _: False,
            samples_io=SamplesIO(target_dir=output_dir, writer=writer),
            save_labels_func=lambda data_type: data_type != DataType.Test,
            sample_rows_provider=sample_row_provider)
    ])

    pipeline.run(input_data=None,
                 params_dict={
                     "data_folding": data_folding,
                     "data_type_pipelines": data_type_pipelines
                 })
