from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from sources.s_rusentrel import do_serialize_bert

do_serialize_bert(writer=PandasCsvWriter(write_header=True))
