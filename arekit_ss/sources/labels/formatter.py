from arekit.common.labels.str_fmt import StringLabelsFormatter

from arekit_ss.sources.labels.sentiment import PositiveTo, NegativeTo


class PosNegLabelsFormatter(StringLabelsFormatter):

    def __init__(self):
        stol = {
            "negative": PositiveTo,
            "positive": NegativeTo
        }
        super(PosNegLabelsFormatter, self).__init__(stol=stol)