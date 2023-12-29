from arekit_ss.sources.rusentrel.utils.const import POS_LABEL_STR, NEG_LABEL_STR
from arekit_ss.sources.rusentrel.utils.io_utils import RuSentRelVersions, RuSentRelIOUtils
from arekit_ss.sources.rusentrel.utils.labels_fmt import RuSentRelLabelsFormatter
from arekit_ss.sources.rusentrel.utils.opinions.provider import RuSentRelOpinionCollectionProvider


class RuSentRelOpinions:
    """
    Collection of sentiment opinions between entities
    """

    @staticmethod
    def iter_from_doc(doc_id, labels_fmt, version=RuSentRelVersions.V11):
        """ doc_id:
            synonyms: None or SynonymsCollection
                None corresponds to the related synonym collection from RuSentRel collection.
            version: RuSentrelVersions enum
        """
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(labels_fmt, RuSentRelLabelsFormatter))
        assert(labels_fmt.supports_value(POS_LABEL_STR))
        assert(labels_fmt.supports_value(NEG_LABEL_STR))

        return RuSentRelIOUtils.iter_from_zip(
            inner_path=RuSentRelIOUtils.get_sentiment_opin_filepath(index=doc_id, version=version),
            process_func=lambda input_file: RuSentRelOpinionCollectionProvider._iter_opinions_from_file(
                input_file=input_file,
                labels_formatter=labels_fmt,
                error_on_non_supported=True),
            version=version)
