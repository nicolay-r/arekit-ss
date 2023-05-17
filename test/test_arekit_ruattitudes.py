import unittest
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

from framework.arekit.bert_sampler import create_bert_sampler
from framework.arekit.serialize_bert import serialize_bert_pipeline
from framework.arekit.serialize_nn import serialize_nn_pipeline
from sources.config import SourcesConfig
from sources.s_rusentrel import build_datapipeline_bert, build_datapipeline_nn


class TestRuAttitudes(unittest.TestCase):

    __output_dir = "_out/"

    def __config(self):
        cfg = SourcesConfig()
        cfg.docs_limit = 5
        cfg.dest_lang = "en"
        return cfg

    def test_serialize_bert_opennre(self):
        data_folding, pipelines = build_datapipeline_bert(self.__config())
        s_ppl = serialize_bert_pipeline(output_dir="_out/ra-bert",
                                        writer=OpenNREJsonWriter(text_columns=["text_a"]),
                                        sample_row_provider=create_bert_sampler(100))
        s_ppl.run(input_data=None,
                  params_dict={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  })

    def test_serialize_nn_csv(self):
        data_folding, pipelines = build_datapipeline_nn(self.__config())
        s_ppl = serialize_nn_pipeline(output_dir="_out/ra-nn",
                                      writer=OpenNREJsonWriter(text_columns=["text_a"]))
        s_ppl.run(input_data=None,
                  params_dict={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  })

    def test_serialize_nn_opennre(self):
        data_folding, pipelines = build_datapipeline_nn(self.__config())
        s_ppl = serialize_nn_pipeline(writer=OpenNREJsonWriter(text_columns=["text_a"]),
                                      output_dir="_out/ra-nn")

        s_ppl.run(input_data=None,
                  params_dict={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  })
