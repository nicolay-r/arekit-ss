from arekit.contrib.bert.input.providers.cropped_sample import CroppedBertSampleRowProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.utils.entities.formatters.str_display import StringEntitiesDisplayValueFormatter


def create_bert_rows_provider(terms_per_context, labels_scaler):
    """Default BERT label scaler"""
    return CroppedBertSampleRowProvider(
        crop_window_size=terms_per_context,
        label_scaler=labels_scaler,
        text_b_template=None,
        text_terms_mapper=BertDefaultStringTextTermsMapper(
            entity_formatter=StringEntitiesDisplayValueFormatter()
        ))
