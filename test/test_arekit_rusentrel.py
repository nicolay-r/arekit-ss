import unittest
from os.path import join

from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions, RuSentRelIOUtils
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.sentinerel.labels import PositiveTo, NegativeTo
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.pipelines.sources.rusentrel.extract_text_opinions import \
    create_text_opinion_extraction_pipeline

from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler
from framework.arekit.serialize_bert import serialize_bert, CroppedBertSampleRowProvider
from framework.arekit.serialize_nn import serialize_nn


class RuSentRelEntitiesFormatter(StringEntitiesFormatter):
    """ Форматирование сущностей. Было принято решение использовать тип сущности в качетстве значений.
        Поскольку тексты русскоязычные, то и типы были руссифицированы из соображений более удачных embeddings.
    """

    type_formatter = {
        "GEOPOLIT": "гео-сущность",
        "ORG": "организация",
        "PER": "личность",
        "LOC": "локация",
        "ОRG": "организация"
    }

    def __init__(self, subject_fmt='[субъект]', object_fmt="[объект]"):
        self.__subject_fmt = subject_fmt
        self.__object_fmt = object_fmt

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        assert(isinstance(entity_type, OpinionEntityType))

        if entity_type == OpinionEntityType.Other:
            return RuSentRelEntitiesFormatter.type_formatter[original_value.Type]
        elif entity_type == OpinionEntityType.Object or entity_type == OpinionEntityType.SynonymObject:
            return self.__object_fmt
        elif entity_type == OpinionEntityType.Subject or entity_type == OpinionEntityType.SynonymSubject:
            return self.__subject_fmt

        return None


class RuSentRelTypedEntitiesFormatter(StringEntitiesFormatter):

    type_formatter = {
        "GEOPOLIT": "гео-сущность",
        "ORG": "организация",
        "PER": "личность",
        "LOC": "локация",
        "ОRG": "организация"
    }

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        return self.type_formatter[original_value.Type]


class TestRuSentRel(unittest.TestCase):
    """ TODO: This might be a test example for AREkit (utils).
    """

    def __test_serialize_bert(self, writer):

        version = RuSentRelVersions.V11

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer()])

        pipeline = create_text_opinion_extraction_pipeline(
            rusentrel_version=version,
            text_parser=text_parser,
            labels_fmt=RuSentRelLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))

        data_folding = NoFolding(doc_ids=RuSentRelIOUtils.iter_collection_indices(version),
                                 supported_data_type=DataType.Train)

        sample_row_provider = CroppedBertSampleRowProvider(
            crop_window_size=50,
            label_scaler=PosNegNeuRelationsLabelScaler(),
            text_b_template=BertTextBTemplates.NLI.value,
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=RuSentRelTypedEntitiesFormatter()
            ))

        serialize_bert(output_dir="_out/serialize-rusentrel-bert",
                       terms_per_context=50,
                       split_filepath=None,
                       data_type_pipelines={DataType.Train: pipeline},
                       sample_row_provider=sample_row_provider,
                       folding_type=None,
                       data_folding=data_folding,
                       writer=writer)

    def __test_serialize_nn(self, writer):

        version = RuSentRelVersions.V11

        stemmer = MystemWrapper()
        frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))
        frame_variant_collection = FrameVariantsCollection()
        frame_variant_collection.fill_from_iterable(
            variants_with_id=frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer(keep_tokens=True),
                                               LemmasBasedFrameVariantsParser(
                                                   frame_variants=frame_variant_collection,
                                                   stemmer=stemmer)])

        pipeline = create_text_opinion_extraction_pipeline(
            rusentrel_version=version,
            text_parser=text_parser,
            labels_fmt=RuSentRelLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))

        data_folding = NoFolding(doc_ids=RuSentRelIOUtils.iter_collection_indices(version),
                                 supported_data_type=DataType.Train)

        serialize_nn(output_dir="_out/serialize-rusentrel-nn",
                     split_filepath=None,
                     data_type_pipelines={DataType.Train: pipeline},
                     folding_type=None,
                     data_folding=data_folding,
                     writer=writer)

    def test_serialize_bert_csv(self):
        self.__test_serialize_bert(writer=PandasCsvWriter(write_header=True))

    def test_serialize_bert_opennre(self):
        self.__test_serialize_bert(writer=OpenNREJsonWriter(text_columns=["text_a", "text_b"]))

    def test_serialize_nn_csv(self):
        self.__test_serialize_nn(writer=PandasCsvWriter(write_header=True))

    def test_serialize_nn_opennre(self):
        self.__test_serialize_nn(writer=OpenNREJsonWriter(text_columns=["text_a"]))
