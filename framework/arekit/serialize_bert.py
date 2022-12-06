from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider

from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.bert import BertExperimentInputSerializerPipelineItem
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer

from SentiNEREL.doc_ops import CollectionDocOperation
from SentiNEREL.folding.factory import FoldingFactory
from SentiNEREL.labels.formatter import SentimentLabelFormatter
from SentiNEREL.pipelines.data import prepare_data_pipelines


class CroppedBertSampleRowProvider(BaseSampleRowProvider):
    """ Нужно немного изменить базовый провайдер так, чтобы
        возвращался небольшой контекст, который влкючает в себя
        объект и субъект, чтобы когда на вход в BERT будут подаваться семплы,
        не возникло проблемы отсечения ввиду огромного предложения.
    """

    def __init__(self, crop_window_size, label_scaler, text_terms_mapper, text_b_template):

        text_provider = BaseSingleTextProvider(text_terms_mapper=text_terms_mapper) \
            if text_b_template is None else PairTextProvider(text_b_template=text_b_template,
                                                             text_terms_mapper=text_terms_mapper)

        super(CroppedBertSampleRowProvider, self).__init__(label_provider=MultipleLabelProvider(label_scaler),
                                                           text_provider=text_provider)

        self.__crop_window_size = crop_window_size

    @staticmethod
    def __calc_window_bounds(window_size, s_ind, t_ind, input_length):
        """ returns: [_from, _to)
        """
        assert(isinstance(s_ind, int))
        assert(isinstance(t_ind, int))
        assert(isinstance(input_length, int))
        assert(input_length >= s_ind and input_length >= t_ind)

        def __in():
            return _from <= s_ind < _to and _from <= t_ind < _to

        _from = 0
        _to = window_size
        while not __in():
            _from += 1
            _to += 1

        return _from, _to

    def _provide_sentence_terms(self, parsed_news, sentence_ind, s_ind, t_ind):
        terms_iter, src_ind, tgt_ind = super(CroppedBertSampleRowProvider, self)._provide_sentence_terms(
            parsed_news=parsed_news, sentence_ind=sentence_ind, s_ind=s_ind, t_ind=t_ind)
        terms = list(terms_iter)
        _from, _to = self.__calc_window_bounds(window_size=self.__crop_window_size,
                                               s_ind=s_ind, t_ind=t_ind, input_length=len(terms))
        return terms[_from:_to], src_ind - _from, tgt_ind - _from


def serialize_bert(split_filepath, terms_per_context, writer, sample_row_provider, output_dir,
                   data_type_pipelines=None, data_folding=None, folding_type="fixed", limit=None):
    assert(isinstance(limit, int) or limit is None)
    assert(isinstance(split_filepath, str) or split_filepath is None)
    assert(isinstance(terms_per_context, int))
    assert(isinstance(sample_row_provider, BaseSampleRowProvider))
    assert(isinstance(output_dir, str))

    pipeline = BasePipeline([
        BertExperimentInputSerializerPipelineItem(
            balance_func=lambda data_type: data_type == DataType.Train,
            samples_io=SamplesIO(target_dir=output_dir, writer=writer),
            save_labels_func=lambda data_type: data_type != DataType.Test,
            sample_rows_provider=sample_row_provider)
    ])

    doc_ops = None

    if data_folding is None:
        # Selecting from presets.
        if folding_type == "fixed":
            filenames_by_ids, data_folding = FoldingFactory.create_fixed_folding(
                fixed_split_filepath=split_filepath, limit=limit)
            doc_ops = CollectionDocOperation(filenames_by_ids)

    if data_type_pipelines is None:
        # considering a default pipeline.
        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer()])
        data_type_pipelines = prepare_data_pipelines(text_parser=text_parser,
                                                     doc_ops=doc_ops,
                                                     label_formatter=SentimentLabelFormatter(),
                                                     terms_per_context=terms_per_context)

    pipeline.run(input_data=None,
                 params_dict={
                     "data_folding": data_folding,
                     "data_type_pipelines": data_type_pipelines
                 })
