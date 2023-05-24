import unittest

from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

from framework.arekit.rows_bert import create_bert_rows_provider
from framework.arekit.rows_nn import create_nn_rows_provider
from framework.arekit.serialize_bert import serialize_bert_pipeline
from framework.arekit.serialize_nn import serialize_nn_pipeline
from sources.config import SourcesConfig
from sources.s_rusentrel import build_datapipeline_bert, build_datapipeline_nn
from sources.scaler import PosNegNeuRelationsLabelScaler


class TestRuAttitudes(unittest.TestCase):

    __output_dir = "_out/"

    def __config(self):
        cfg = SourcesConfig()
        cfg.docs_limit = 5
        cfg.dest_lang = "en"
        return cfg

    def test_serialize_bert_opennre(self):
        data_folding, pipelines = build_datapipeline_bert(self.__config())
        item = serialize_bert_pipeline(output_dir="_out/ra-bert",
                                       writer=OpenNREJsonWriter(text_columns=["text_a"]),
                                       rows_provider=create_bert_rows_provider(
                                           terms_per_context=100,
                                           labels_scaler=PosNegNeuRelationsLabelScaler()))
        s_ppl = BasePipeline([item])
        s_ppl.run(input_data=None,
                  params_dict={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  })

    def test_serialize_nn_csv(self):
        data_folding, pipelines = build_datapipeline_nn(self.__config())
        item = serialize_nn_pipeline(output_dir="_out/ra-nn",
                                     writer=OpenNREJsonWriter(text_columns=["text_a"]),
                                     rows_provider=create_nn_rows_provider(
                                         labels_scaler=PosNegNeuRelationsLabelScaler()))
        s_ppl = BasePipeline([item])
        s_ppl.run(input_data=None,
                  params_dict={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  })

    def test_serialize_nn_opennre(self):
        data_folding, pipelines = build_datapipeline_nn(self.__config())
        item = serialize_nn_pipeline(writer=OpenNREJsonWriter(text_columns=["text_a"]),
                                     output_dir="_out/ra-nn",
                                     rows_provider=create_nn_rows_provider(
                                         labels_scaler=PosNegNeuRelationsLabelScaler()))

        s_ppl = BasePipeline([item])
        s_ppl.run(input_data=None,
                  params_dict={
                      "data_folding": data_folding,
                      "data_type_pipelines": pipelines
                  })
