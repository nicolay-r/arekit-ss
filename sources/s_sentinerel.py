from arekit.common.experiment.data_type import DataType
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesEffectLabelsFormatter, \
    RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.sentinerel.io_utils import SentiNerelVersions
from arekit.contrib.source.sentinerel.labels import PositiveTo, NegativeTo
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.sources.sentinerel.extract_text_opinions import \
    create_text_opinion_extraction_pipeline
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from sources.config import SourcesConfig
from sources.processing.translator import TextAndEntitiesGoogleTranslator


def build_datapipeline_bert(cfg):
    assert(isinstance(cfg, SourcesConfig))

    text_parser = BaseTextParser(pipeline=[
        BratTextEntitiesParser(),
        TextAndEntitiesGoogleTranslator(src="ru", dest=cfg.dest_lang) if cfg.dest_lang != 'ru' else None,
        DefaultTextTokenizer()])

    pipelines, data_folding = create_text_opinion_extraction_pipeline(
        sentinerel_version=SentiNerelVersions.V21,
        terms_per_context=cfg.terms_per_context,
        docs_limit=cfg.docs_limit,
        doc_ops=None,
        text_parser=text_parser)

    return data_folding, {DataType.Train: pipelines[DataType.Train]}


def build_datapipeline_nn(cfg):
    assert(isinstance(cfg, SourcesConfig))

    stemmer = MystemWrapper()

    frames_collection = RuSentiFramesCollection.read(
        version=RuSentiFramesVersions.V20,
        labels_fmt=RuSentiFramesLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo),
        effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))
    frame_variant_collection = FrameVariantsCollection()
    frame_variant_collection.fill_from_iterable(
        variants_with_id=frames_collection.iter_frame_id_and_variants(),
        overwrite_existed_variant=True,
        raise_error_on_existed_variant=False)

    text_parser = BaseTextParser(pipeline=[
        BratTextEntitiesParser(),
        DefaultTextTokenizer(keep_tokens=True),
        TextAndEntitiesGoogleTranslator(src="ru", dest=cfg.dest_lang) if cfg.dest_lang != 'ru' else None,
        LemmasBasedFrameVariantsParser(
            frame_variants=frame_variant_collection,
            stemmer=stemmer)])

    pipelines, data_folding = create_text_opinion_extraction_pipeline(
        sentinerel_version=SentiNerelVersions.V21,
        text_parser=text_parser,
        terms_per_context=cfg.terms_per_context,
        docs_limit=cfg.docs_limit,
        doc_ops=None)

    return data_folding, {DataType.Train: pipelines[DataType.Train]}
