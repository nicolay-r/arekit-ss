from arekit.common.text.parser import BaseTextParser
from arekit_ss.sources.config import SourcesConfig


def create_lm(cfg):
    assert(isinstance(cfg, SourcesConfig))

    return BaseTextParser(pipeline=[
        cfg.entities_parser,
        cfg.get_translator_pipeline_item(do_translation=cfg.src_lang != cfg.dest_lang),
    ])
