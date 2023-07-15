from arekit.common.experiment.data_type import DataType
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.base import BaseWriter
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.networks import NetworksInputSerializerPipelineItem


def serialize_nn_pipeline(output_dir, writer, rows_provider):
    """ Run data preparation process for neural networks, i.e.
        convolutional neural networks and recurrent-based neural networks.
        Implementation based on AREkit toolkit API.
    """
    assert(isinstance(output_dir, str))
    assert(isinstance(writer, BaseWriter))

    return NetworksInputSerializerPipelineItem(
        storage=RowCacheStorage(),
        rows_provider=rows_provider,
        samples_io=SamplesIO(target_dir=output_dir, writer=writer),
        emb_io=NpEmbeddingIO(target_dir=output_dir),
        balance_func=lambda _: False,
        save_labels_func=lambda data_type: data_type != DataType.Test,
        save_embedding=True)