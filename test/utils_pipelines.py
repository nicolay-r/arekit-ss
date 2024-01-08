from os.path import dirname, join, realpath

from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO

from arekit_ss.core.rows_bert import create_bert_rows_provider
from arekit_ss.core.rows_ru_sentiment_nn import create_ru_sentiment_nn_rows_provider
from arekit_ss.core.samples_io import CustomSamplesIO
from arekit_ss.core.serialize_bert import serialize_bert_pipeline
from arekit_ss.core.serialize_nn import serialize_nn_pipeline
from arekit_ss.entity.masking import StringEntitiesDisplayValueFormatter
from arekit_ss.sources.labels.scaler import PosNegNeuRelationsLabelScaler
from arekit_ss.sources.labels.scaler_frames import ThreeLabelScaler

current_dir = dirname(realpath(__file__))
TEST_DATA_DIR = join(current_dir, "data")
TEST_OUT_DIR = join("_out")


def nn_ppl(collection_name):
    return serialize_nn_pipeline(
        samples_io=CustomSamplesIO(target_dir=TEST_OUT_DIR,
                                   writer=OpenNREJsonWriter(text_columns=["text_a"]),
                                   prefix=collection_name),
        emb_io=NpEmbeddingIO(target_dir=TEST_OUT_DIR, prefix_name="-".join(["nn", collection_name])),
        rows_provider=create_ru_sentiment_nn_rows_provider(relation_labels_scaler=PosNegNeuRelationsLabelScaler(),
                                                           frame_roles_label_scaler=ThreeLabelScaler(),
                                                           vectorizers="default",
                                                           entity_fmt=StringEntitiesDisplayValueFormatter()))


def bert_ppl(collection_name):
    return serialize_bert_pipeline(
        samples_io=CustomSamplesIO(target_dir=TEST_OUT_DIR,
                                   writer=OpenNREJsonWriter(["text_a"]),
                                   prefix="-".join(["bert", collection_name])),
        rows_provider=create_bert_rows_provider(terms_per_context=100,
                                                labels_scaler=PosNegNeuRelationsLabelScaler(),
                                                entity_fmt=StringEntitiesDisplayValueFormatter()))
