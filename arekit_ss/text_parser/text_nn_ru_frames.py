from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from arekit_ss.sources.config import SourcesConfig
from arekit_ss.sources.labels.sentiment import PositiveTo, NegativeTo
from arekit_ss.text_parser.translator import TextAndEntitiesGoogleTranslator


def create_nn_ru_frames(cfg):
    """ This pipeline involves an application of the RuSentiFrames.
        The latter represent a lexicon for texts in Russian.
    """
    assert(isinstance(cfg, SourcesConfig))
    assert(cfg.src_lang == "ru")

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

    return BaseTextParser(pipeline=[
        cfg.entities_parser,
        DefaultTextTokenizer(keep_tokens=True),
        TextAndEntitiesGoogleTranslator(src=cfg.src_lang, dest=cfg.dest_lang) if cfg.dest_lang != cfg.src_lang else None,
        LemmasBasedFrameVariantsParser(frame_variants=frame_variant_collection, stemmer=stemmer)])
