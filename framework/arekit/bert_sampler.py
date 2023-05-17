from arekit.contrib.bert.input.providers.cropped_sample import CroppedBertSampleRowProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.utils.entities.formatters.str_display import StringEntitiesDisplayValueFormatter

from sources.scaler import PosNegNeuRelationsLabelScaler


def create_bert_sampler(terms_per_context):
    """Default BERT label scaler"""
    return CroppedBertSampleRowProvider(
        crop_window_size=terms_per_context,
        label_scaler=PosNegNeuRelationsLabelScaler(),
        text_b_template=None,
        text_terms_mapper=BertDefaultStringTextTermsMapper(
            entity_formatter=StringEntitiesDisplayValueFormatter()
        ))
