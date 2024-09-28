import logging
import unittest

from arekit.common.data.input.providers.const import IDLE_MODE
from arekit.common.docs.base import Document
from arekit.common.docs.parser import DocumentParsers
from arekit.common.docs.sentence import BaseDocumentSentence
from arekit.common.entities.base import Entity
from arekit.common.context.token import Token
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.pipelines.items.text.entities_default import TextEntitiesParser

from arekit_ss.pipelines.text.tokenizer import DefaultTextTokenizer
from arekit_ss.sources.config import SourcesConfig
from arekit_ss.third_party.googletrans import translate_value

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


class TestTextParser(unittest.TestCase):

    def test_translator(self):
        x = translate_value("привет", dest="en", src="ru")
        print(x)

    def test(self):
        text = "А контроль над этими провинциями — [США] , которая не пытается ввести санкции против. [ВКC] "

        cfg = SourcesConfig()
        cfg.src_lang = "ru"
        cfg.dest_lang = "en"

        # Adopting translate pipeline item, based on google translator.
        pipeline_items = [TextEntitiesParser(src_func=lambda s: s.Text),
                          cfg.get_translator_pipeline_item(cfg.src_lang != cfg.dest_lang),
                          DefaultTextTokenizer(keep_tokens=True)]

        doc = Document(doc_id=0, sentences=[BaseDocumentSentence(text.split())])
        parsed_doc = DocumentParsers.parse(doc=doc, pipeline_items=pipeline_items,
                                           parent_ppl_ctx=PipelineContext({IDLE_MODE: False}))
        self.debug_show_terms(parsed_doc.iter_terms())

    @staticmethod
    def debug_show_terms(terms):
        for term in terms:
            if isinstance(term, str):
                print("Word:\t\t'{}'".format(term))
            elif isinstance(term, Token):
                print("Token:\t\t'{}' ('{}')".format(term.get_token_value(), term.get_meta_value()))
            elif isinstance(term, Entity):
                print("Entity:\t\t'{}' ({})".format(term.Value, type(term)))
            else:
                raise Exception("unsupported type {}".format(term))


if __name__ == '__main__':
    unittest.main()