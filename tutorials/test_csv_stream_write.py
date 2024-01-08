import os
import unittest
from collections import OrderedDict
from os.path import dirname, join

from arekit.common.data.doc_provider import DocumentProvider
from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import Label, NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter
from arekit.contrib.utils.pipelines.items.sampling.base import BaseSerializerPipelineItem
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter

from arekit_ss.core.samples_io import CustomSamplesIO
from arekit_ss.core.source.brat.entities.parser import BratTextEntitiesParser
from arekit_ss.pipelines.annot.predefined import PredefinedTextOpinionAnnotator
from tutorials.test_tutorial_collection_binding import FooDocReader
from tutorials.test_tutorial_pipeline_sampling_bert import CustomEntitiesFormatter, CustomLabelsFormatter


class FooDocumentProvider(DocumentProvider):
    def by_id(self, doc_id):
        return FooDocReader.read_document(str(doc_id), doc_id=doc_id)


class Positive(Label):
    pass


class Negative(Label):
    pass


class SentimentLabelScaler(BaseLabelScaler):

    def __init__(self):
        int_to_label = OrderedDict([(NoLabel(), 0), (Positive(), 1), (Negative(), -1)])
        uint_to_label = OrderedDict([(NoLabel(), 0), (Positive(), 1), (Negative(), 2)])
        super(SentimentLabelScaler, self).__init__(int_dict=int_to_label,
                                                   uint_dict=uint_to_label)


class TestStreamWriters(unittest.TestCase):

    __output_dir = join(dirname(__file__), "out")

    def __launch(self, writer):

        text_b_template = '{subject} к {object} в контексте : << {context} >>'

        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir)

        terms_mapper = BertDefaultStringTextTermsMapper(
            entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O"))

        text_provider = BaseSingleTextProvider(terms_mapper) \
            if text_b_template is None else \
            PairTextProvider(text_b_template, terms_mapper)

        sample_rows_provider = BaseSampleRowProvider(
            label_provider=MultipleLabelProvider(SentimentLabelScaler()),
            text_provider=text_provider)

        samples_io = CustomSamplesIO(self.__output_dir, writer)

        pipeline_item = BaseSerializerPipelineItem(
            rows_provider=sample_rows_provider,
            samples_io=samples_io,
            save_labels_func=lambda data_type: True,
            storage=RowCacheStorage())

        #####
        # Declaring pipeline related context parameters.
        #####
        doc_provider = FooDocumentProvider()
        pipeline_items = [BratTextEntitiesParser(),
                          DefaultTextTokenizer(keep_tokens=True)]
        train_pipeline = text_opinion_extraction_pipeline(
            annotators=[
                PredefinedTextOpinionAnnotator(
                    doc_provider,
                    label_formatter=CustomLabelsFormatter(pos_label_type=Positive,
                                                          neg_label_type=Negative))
            ],
            text_opinion_filters=[
                DistanceLimitedTextOpinionFilter(terms_per_context=50)
            ],
            get_doc_by_id_func=doc_provider.by_id,
            entity_index_func=lambda brat_entity: brat_entity.ID,
            pipeline_items=pipeline_items)
        #####

        BasePipelineLauncher.run(
            pipeline=[pipeline_item],
            pipeline_ctx=PipelineContext(d={
                "data_type_pipelines": {DataType.Train: train_pipeline},
                "data_folding": {DataType.Train: [0, 1]}
            }),
            has_input=False)

    def test_csv_native(self):
        """ Testing writing into CSV format
        """
        return self.__launch(writer=NativeCsvWriter())

    def test_json_native(self):
        """ Testing writing into CSV format
        """
        return self.__launch(writer=OpenNREJsonWriter(text_columns=[BaseSingleTextProvider.TEXT_A]))
