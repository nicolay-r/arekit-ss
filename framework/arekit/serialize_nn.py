from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.core.input.term_types import TermTypes
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.sentinerel.labels import PositiveTo, NegativeTo
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider
from arekit.contrib.utils.data.writers.base import BaseWriter
from arekit.contrib.utils.entities.formatters.str_display import StringEntitiesDisplayValueFormatter
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.networks import NetworksInputSerializerPipelineItem
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.contrib.utils.resources import load_embedding_news_mystem_skipgram_1000_20_2015
from arekit.contrib.utils.vectorizers.bpe import BPEVectorizer
from arekit.contrib.utils.vectorizers.random_norm import RandomNormalVectorizer

from sources.scaler import PosNegNeuRelationsLabelScaler


def serialize_nn(output_dir, writer,
                 labels_scaler=PosNegNeuRelationsLabelScaler(),
                 entities_fmt=StringEntitiesDisplayValueFormatter(),
                 data_folding=None, data_type_pipelines=None, limit=None, suffix="nn"):
    """ Run data preparation process for neural networks, i.e.
        convolutional neural networks and recurrent-based neural networks.
        Implementation based on AREkit toolkit API.
    """
    assert(isinstance(suffix, str))
    assert(isinstance(output_dir, str))
    assert(isinstance(limit, int) or limit is None)
    assert(isinstance(writer, BaseWriter))

    stemmer = MystemWrapper()
    pos_tagger = POSMystemWrapper(mystem=stemmer.MystemInstance)

    # Frames initialization
    frames_collection = RuSentiFramesCollection.read_collection(
        version=RuSentiFramesVersions.V20,
        labels_fmt=RuSentiFramesLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo),
        effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))
    frames_connotation_provider = RuSentiFramesConnotationProvider(frames_collection)

    ctx = NetworkSerializationContext(
        labels_scaler=labels_scaler,
        pos_tagger=pos_tagger,
        frame_roles_label_scaler=ThreeLabelScaler(),
        frames_connotation_provider=frames_connotation_provider)

    embedding = load_embedding_news_mystem_skipgram_1000_20_2015(stemmer)
    bpe_vectorizer = BPEVectorizer(embedding=embedding, max_part_size=3)
    norm_vectorizer = RandomNormalVectorizer(vector_size=embedding.VectorSize,
                                             token_offset=12345)

    pipeline_item = NetworksInputSerializerPipelineItem(
        vectorizers={
            TermTypes.WORD: bpe_vectorizer,
            TermTypes.ENTITY: bpe_vectorizer,
            TermTypes.FRAME: bpe_vectorizer,
            TermTypes.TOKEN: norm_vectorizer
        },
        samples_io=SamplesIO(target_dir=output_dir, writer=writer),
        emb_io=NpEmbeddingIO(target_dir=output_dir),
        str_entity_fmt=entities_fmt,
        balance_func=lambda data_type: data_type == DataType.Train,
        save_labels_func=lambda data_type: data_type != DataType.Test,
        ctx=ctx,
        save_embedding=True)

    doc_ops = None

    ppl = BasePipeline([pipeline_item])
    ppl.run(input_data=None,
            params_dict={
                "data_folding": data_folding,
                "data_type_pipelines": data_type_pipelines
    })
