from arekit.common.text.parser import BaseTextParser
from arekit_ss.sources.config import SourcesConfig
from arekit_ss.text_parser.translator import TextAndEntitiesGoogleTranslator


def create_lm(cfg):
    assert(isinstance(cfg, SourcesConfig))

    return BaseTextParser(pipeline=[
        cfg.entities_parser,
        TextAndEntitiesGoogleTranslator(src=cfg.src_lang, dest=cfg.dest_lang) if cfg.dest_lang != cfg.src_lang else None,
        ])
