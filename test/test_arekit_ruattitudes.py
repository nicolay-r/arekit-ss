import unittest

from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser

from arekit_ss.sources.config import SourcesConfig
from arekit_ss.sources.ruattitudes.data_pipeline import build_ruattitudes_datapipeline
from arekit_ss.text_parser.text_lm import create_lm
from arekit_ss.text_parser.text_nn_ru_frames import create_nn_ru_frames
from utils import bert_ppl, nn_ppl


class TestRuAttitudes(unittest.TestCase):

    def __config(self):
        cfg = SourcesConfig()
        cfg.docs_limit = 5
        cfg.src_lang = "ru"
        cfg.dest_lang = "ru"
        return cfg

    def test_serialize_bert_opennre(self):
        cfg = self.__config()
        cfg.entities_parser = RuAttitudesTextEntitiesParser()
        cfg.text_parser = create_lm(cfg)
        data_folding, pipelines = build_ruattitudes_datapipeline(cfg)
        s_ppl = BasePipeline([bert_ppl("ra")])
        s_ppl.run(input_data=PipelineContext(d={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  }))

    def test_serialize_nn_csv(self):
        cfg = self.__config()
        cfg.entities_parser = RuAttitudesTextEntitiesParser()
        cfg.text_parser = create_nn_ru_frames(cfg)
        data_folding, pipelines = build_ruattitudes_datapipeline(cfg)
        s_ppl = BasePipeline([nn_ppl("ra")])
        s_ppl.run(input_data=PipelineContext(d={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  }))

    def test_serialize_nn_opennre(self):
        cfg = self.__config()
        cfg.entities_parser = RuAttitudesTextEntitiesParser()
        cfg.text_parser = create_nn_ru_frames(cfg)
        data_folding, pipelines = build_ruattitudes_datapipeline(cfg)
        s_ppl = BasePipeline([nn_ppl("ra")])
        s_ppl.run(input_data=None,
                  params_dict=PipelineContext(d={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  }))
