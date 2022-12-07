import unittest
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter


class TestRuAttitudes(unittest.TestCase):

    def test_serialize_bert_opennre(self):
        self.__test_serialize_bert(writer=OpenNREJsonWriter(text_columns=["text_a", "text_b"]))

    def test_serialize_nn_csv(self):
        self.__test_serialize_nn(writer=PandasCsvWriter(write_header=True))

    def test_serialize_nn_opennre(self):
        self.__test_serialize_nn(writer=OpenNREJsonWriter(text_columns=["text_a"]))
