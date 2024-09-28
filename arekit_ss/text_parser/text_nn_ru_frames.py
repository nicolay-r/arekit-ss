from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from arekit_ss.pipelines.text.tokenizer import DefaultTextTokenizer
from arekit_ss.sources.config import SourcesConfig
from arekit_ss.sources.labels.sentiment import PositiveTo, NegativeTo
from arekit_ss.sources.rusentiframes.collection import RuSentiFramesCollection
from arekit_ss.sources.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, RuSentiFramesEffectLabelsFormatter
from arekit_ss.sources.rusentiframes.types import RuSentiFramesVersions


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

    return [cfg.entities_parser,
            DefaultTextTokenizer(keep_tokens=True),
            cfg.get_translator_pipeline_item(do_translation=cfg.src_lang != cfg.dest_lang),
            LemmasBasedFrameVariantsParser(frame_variants=frame_variant_collection, stemmer=stemmer)]
