from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.sentinerel.labels import PositiveTo, NegativeTo
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.base import BaseWriter
from arekit.contrib.utils.entities.formatters.str_display import StringEntitiesDisplayValueFormatter
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.networks import NetworksInputSerializerPipelineItem
from arekit.contrib.networks.input.ctx_serialization import NetworkSerializationContext

from sources.scaler import PosNegNeuRelationsLabelScaler
from sources.scaler_frames import ThreeLabelScaler


def serialize_nn_pipeline(output_dir, writer,
                          labels_scaler=PosNegNeuRelationsLabelScaler(),
                          entities_fmt=StringEntitiesDisplayValueFormatter()):
    """ Run data preparation process for neural networks, i.e.
        convolutional neural networks and recurrent-based neural networks.
        Implementation based on AREkit toolkit API.
    """
    assert(isinstance(output_dir, str))
    assert(isinstance(writer, BaseWriter))

    # Frames initialization
    frames_collection = RuSentiFramesCollection.read(
        version=RuSentiFramesVersions.V20,
        labels_fmt=RuSentiFramesLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo),
        effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))
    frames_connotation_provider = RuSentiFramesConnotationProvider(frames_collection)

    ctx = NetworkSerializationContext(
        labels_scaler=labels_scaler,
        frame_roles_label_scaler=ThreeLabelScaler(),
        frames_connotation_provider=frames_connotation_provider)

    pipeline_item = NetworksInputSerializerPipelineItem(
        vectorizers=None,
        storage=RowCacheStorage(),
        samples_io=SamplesIO(target_dir=output_dir, writer=writer),
        emb_io=NpEmbeddingIO(target_dir=output_dir),
        str_entity_fmt=entities_fmt,
        balance_func=lambda _: False,
        save_labels_func=lambda data_type: data_type != DataType.Test,
        ctx=ctx,
        save_embedding=True)

    return BasePipeline([pipeline_item])

