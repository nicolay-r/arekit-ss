from arekit.common.data.const import ENTITIES, ENTITY_TYPES
from arekit.contrib.networks.input.const import FrameVariantIndices, FrameConnotations
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.pipelines.items.sampling.networks import NetworksInputSerializerPipelineItem


def serialize_nn_pipeline(samples_io, emb_io, rows_provider):
    """ Run data preparation process for neural networks, i.e.
        convolutional neural networks and recurrent-based neural networks.
        Implementation based on AREkit toolkit API.
    """

    return NetworksInputSerializerPipelineItem(
        storage=RowCacheStorage(force_collect_columns=[
            FrameVariantIndices, FrameConnotations, ENTITIES, ENTITY_TYPES]),
        rows_provider=rows_provider,
        samples_io=samples_io,
        emb_io=emb_io,
        save_labels_func=lambda _: True,
        save_embedding=True)
