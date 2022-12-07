from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesEffectLabelsFormatter, \
    RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.sentinerel.labels import PositiveTo, NegativeTo
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.sources.ruattitudes.extract_text_opinions import \
    create_text_opinion_extraction_pipeline
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from framework.arekit.serialize_bert import CroppedBertSampleRowProvider, serialize_bert
from framework.arekit.serialize_nn import serialize_nn
from sources.helper import EntityHelper
from sources.scaler import PosNegNeuRelationsLabelScaler

from translator import TextAndEntitiesGoogleTranslator


class RuAttitudesEntitiesFormatter(StringEntitiesFormatter):
    """ Форматирование сущностей. Было принято решение использовать тип сущности в качетстве значений.
        Поскольку тексты русскоязычные, то и типы были руссифицированы из соображений более удачных embeddings.
    """

    type_formatter = {
        "GPE": "гео-сущность",
        "PERSON": "личность",
        "LOCAL": "локация",
        "ОRG": "организация"
    }

    def __init__(self, subject_fmt='[субъект]', object_fmt="[объект]"):
        self.__subject_fmt = subject_fmt
        self.__object_fmt = object_fmt

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        assert(isinstance(entity_type, OpinionEntityType))

        if entity_type == OpinionEntityType.Other:
            return self.type_formatter[original_value.Type] \
                if original_value.Type in self.type_formatter else original_value.Value
        elif entity_type == OpinionEntityType.Object or entity_type == OpinionEntityType.SynonymObject:
            return self.__object_fmt
        elif entity_type == OpinionEntityType.Subject or entity_type == OpinionEntityType.SynonymSubject:
            return self.__subject_fmt

        return None


class RuAttitudesTypedEntitiesFormatter(StringEntitiesFormatter):
    """ Форматирование сущностей. Было принято решение использовать тип сущности в качетстве значений.
        Поскольку тексты русскоязычные, то и типы были руссифицированы из соображений более удачных embeddings.
    """

    fmts = {
        'QUANTITY': "количество",
        'ORG': "организация",
        'LAW': "закон",
        'FAC': "сооружение",
        'PERCENT': "процент",
        'NORP': "связь",
        'GPE': "гео-сущность",
        'CARDINAL': "число",
        'LOC': "локация",
    }

    def __init__(self):
        self.__st = set()

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        return self.fmts[original_value.Type] if original_value.Type in self.fmts else \
            EntityHelper.format(original_value)


def __test_serialize_bert(writer):
    text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                           TextAndEntitiesGoogleTranslator(src="ru", dest="en"),
                                           DefaultTextTokenizer()])

    pipeline, ru_attitudes = create_text_opinion_extraction_pipeline(
        text_parser=text_parser, label_scaler=PosNegNeuRelationsLabelScaler())

    data_folding = NoFolding(doc_ids=ru_attitudes.keys(), supported_data_type=DataType.Train)

    sample_row_provider = CroppedBertSampleRowProvider(
        crop_window_size=50,
        label_scaler=PosNegNeuRelationsLabelScaler(),
        text_b_template=BertTextBTemplates.NLI.value,
        text_terms_mapper=BertDefaultStringTextTermsMapper(
            entity_formatter=RuAttitudesTypedEntitiesFormatter()
        ))

    serialize_bert(output_dir="_out/serialize-ruattitudes-bert",
                   terms_per_context=50,
                   split_filepath=None,
                   data_type_pipelines={DataType.Train: pipeline},
                   sample_row_provider=sample_row_provider,
                   data_folding=data_folding,
                   writer=writer)


def test_serialize_nn(writer):
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

    text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                           DefaultTextTokenizer(keep_tokens=True),
                                           LemmasBasedFrameVariantsParser(
                                               frame_variants=frame_variant_collection,
                                               stemmer=stemmer)])

    pipeline, ru_attitudes = create_text_opinion_extraction_pipeline(
        text_parser=text_parser, label_scaler=PosNegNeuRelationsLabelScaler())

    data_folding = NoFolding(doc_ids=ru_attitudes.keys(),
                             supported_data_type=DataType.Train)

    serialize_nn(output_dir="_out/serialize-ruattitudes-nn",
                 data_type_pipelines={DataType.Train: pipeline},
                 data_folding=data_folding,
                 writer=writer)
