from arekit_ss.sources.config import SourcesConfig
from arekit_ss.sources.sentinerel.extract_text_opinions import create_text_opinion_extraction_pipeline
from arekit_ss.sources.sentinerel.utils.io_utils import SentiNerelVersions


def build_sentinerel_datapipeline(cfg):
    assert(isinstance(cfg, SourcesConfig))

    pipelines, data_folding = create_text_opinion_extraction_pipeline(
        sentinerel_version=SentiNerelVersions.V21,
        terms_per_context=cfg.terms_per_context,
        custom_text_opinion_filters=cfg.optional_filters,
        docs_limit=cfg.docs_limit,
        doc_provider=None,
        text_parser=cfg.text_parser_items)

    return data_folding, pipelines
