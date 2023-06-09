from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer

from arekit_ss.sources.config import SourcesConfig
from arekit_ss.text_parser.translator import TextAndEntitiesGoogleTranslator


def create_lm(cfg):
    assert(isinstance(cfg, SourcesConfig))

    return BaseTextParser(pipeline=[
        cfg.entities_parser,
        TextAndEntitiesGoogleTranslator(src="ru", dest=cfg.dest_lang) if cfg.dest_lang != 'ru' else None,
        DefaultTextTokenizer()])
