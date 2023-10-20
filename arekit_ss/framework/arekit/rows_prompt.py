from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.prompt.sample import PromptedSampleRowProvider


def create_prompt_rows_provider(prompt, labels_scaler, labels_formatter, entity_fmt):
    return PromptedSampleRowProvider(
        prompt=prompt,
        crop_window_size=200,
        label_scaler=labels_scaler,
        text_provider=BaseSingleTextProvider(
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=entity_fmt)),
        label_fmt=labels_formatter)
