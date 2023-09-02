from arekit.contrib.source.sentinerel.io_utils import SentiNerelVersions
from arekit.contrib.utils.pipelines.sources.sentinerel.extract_text_opinions import \
    create_text_opinion_extraction_pipeline

from arekit_ss.sources.config import SourcesConfig


def build_sentinerel_datapipeline(cfg):
    assert(isinstance(cfg, SourcesConfig))

    pipelines, data_folding = create_text_opinion_extraction_pipeline(
        sentinerel_version=SentiNerelVersions.V21,
        terms_per_context=cfg.terms_per_context,
        docs_limit=cfg.docs_limit,
        doc_provider=None,
        text_parser=cfg.text_parser)

    return data_folding, pipelines
