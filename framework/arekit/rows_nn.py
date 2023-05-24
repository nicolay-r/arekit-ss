from arekit.contrib.networks.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.sentinerel.labels import PositiveTo, NegativeTo
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider
from arekit.contrib.utils.entities.formatters.str_display import StringEntitiesDisplayValueFormatter
from arekit.contrib.utils.nn.rows import create_rows_provider

from sources.scaler_frames import ThreeLabelScaler


def create_nn_rows_provider(labels_scaler):
    """create rows provider for neural networks
    """

    frames_collection = RuSentiFramesCollection.read(
        version=RuSentiFramesVersions.V20,
        labels_fmt=RuSentiFramesLabelsFormatter(
            pos_label_type=PositiveTo, neg_label_type=NegativeTo),
        effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
            pos_label_type=PositiveTo, neg_label_type=NegativeTo))

    ctx = NetworkSerializationContext(
        labels_scaler=labels_scaler,
        frame_roles_label_scaler=ThreeLabelScaler(),
        frames_connotation_provider=RuSentiFramesConnotationProvider(frames_collection))

    return create_rows_provider(str_entity_fmt=StringEntitiesDisplayValueFormatter(), ctx=ctx)
