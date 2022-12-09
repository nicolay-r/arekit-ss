from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser

from arekit.contrib.bert.input.providers.cropped_sample import CroppedBertSampleRowProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesEffectLabelsFormatter, RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.sentinerel.labels import PositiveTo, NegativeTo
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates
from arekit.contrib.utils.entities.formatters.str_display import StringEntitiesDisplayValueFormatter
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.sources.ruattitudes.extract_text_opinions import create_text_opinion_extraction_pipeline
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from framework.arekit.serialize_bert import serialize_bert
from framework.arekit.serialize_nn import serialize_nn

from sources.scaler import PosNegNeuRelationsLabelScaler

from translator import TextAndEntitiesGoogleTranslator


def do_serialize_bert(writer, output_dir, terms_per_context=50, dest_lang="en", limit=None):

    text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                           TextAndEntitiesGoogleTranslator(src="ru", dest=dest_lang),
                                           DefaultTextTokenizer()])

    pipeline, ru_attitudes = create_text_opinion_extraction_pipeline(
        text_parser=text_parser, label_scaler=PosNegNeuRelationsLabelScaler(), limit=limit)

    data_folding = NoFolding(doc_ids=ru_attitudes.keys(), supported_data_type=DataType.Train)

    sample_row_provider = CroppedBertSampleRowProvider(
        crop_window_size=terms_per_context,
        label_scaler=PosNegNeuRelationsLabelScaler(),
        text_b_template=BertTextBTemplates.NLI.value,
        text_terms_mapper=BertDefaultStringTextTermsMapper(
            entity_formatter=StringEntitiesDisplayValueFormatter()
        ))

    serialize_bert(output_dir=output_dir,
                   data_type_pipelines={DataType.Train: pipeline},
                   sample_row_provider=sample_row_provider,
                   data_folding=data_folding,
                   writer=writer)


def do_serialize_nn(writer, output_dir, dest_lang="en", limit=None):

    stemmer = MystemWrapper()

    # Adopt frames annotation.
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
                                           TextAndEntitiesGoogleTranslator(src="ru", dest=dest_lang),
                                           LemmasBasedFrameVariantsParser(
                                               frame_variants=frame_variant_collection,
                                               stemmer=stemmer)])

    pipeline, ru_attitudes = create_text_opinion_extraction_pipeline(
        text_parser=text_parser, label_scaler=PosNegNeuRelationsLabelScaler(), limit=limit)

    data_folding = NoFolding(doc_ids=ru_attitudes.keys(),
                             supported_data_type=DataType.Train)

    serialize_nn(output_dir=output_dir,
                 data_type_pipelines={DataType.Train: pipeline},
                 data_folding=data_folding,
                 writer=writer)
