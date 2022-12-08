import unittest
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter

from sources.s_ruattitudes import do_serialize_bert, do_serialize_nn


class TestRuAttitudes(unittest.TestCase):

    def test_serialize_bert_opennre(self):
        do_serialize_bert(writer=OpenNREJsonWriter(text_columns=["text_a", "text_b"]))

    def test_serialize_nn_csv(self):
        do_serialize_nn(writer=PandasCsvWriter(write_header=True), limit=5)

    def test_serialize_nn_opennre(self):
        do_serialize_nn(writer=OpenNREJsonWriter(text_columns=["text_a"]), limit=5)
