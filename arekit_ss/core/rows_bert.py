from arekit.contrib.bert.input.providers.cropped_sample import CroppedBertSampleRowProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper


def create_bert_rows_provider(terms_per_context, labels_scaler, entity_fmt):
    """Default BERT label scaler"""
    return CroppedBertSampleRowProvider(
        crop_window_size=terms_per_context,
        label_scaler=labels_scaler,
        text_b_template=None,
        text_terms_mapper=BertDefaultStringTextTermsMapper(
            entity_formatter=entity_fmt
        ))
