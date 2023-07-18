from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.contrib.networks.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider
from arekit.contrib.utils.entities.formatters.str_display import StringEntitiesDisplayValueFormatter
from arekit.contrib.utils.nn.rows import create_rows_provider


def create_ru_sentiment_nn_rows_provider(relation_labels_scaler, frame_roles_label_scaler, vectorizers):
    assert(isinstance(relation_labels_scaler, BaseLabelScaler))
    assert(isinstance(frame_roles_label_scaler, SentimentLabelScaler))
    assert(frame_roles_label_scaler.LabelsCount == 3)
    assert(relation_labels_scaler.LabelsCount == 3)

    pos_label = frame_roles_label_scaler.int_to_label(1)
    neg_label = frame_roles_label_scaler.invert_label(pos_label)

    frames_collection = RuSentiFramesCollection.read(
        version=RuSentiFramesVersions.V20,
        labels_fmt=RuSentiFramesLabelsFormatter(
            pos_label_type=type(pos_label), neg_label_type=type(neg_label)),
        effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
            pos_label_type=type(pos_label), neg_label_type=type(neg_label)))

    ctx = NetworkSerializationContext(
        labels_scaler=relation_labels_scaler,
        frame_roles_label_scaler=frame_roles_label_scaler,
        frames_connotation_provider=RuSentiFramesConnotationProvider(frames_collection))

    return create_rows_provider(str_entity_fmt=StringEntitiesDisplayValueFormatter(), ctx=ctx,
                                vectorizers=vectorizers)
