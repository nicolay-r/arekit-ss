from arekit.common.labels.base import NoLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter

from arekit_ss.sources.labels.sentiment import PositiveTo, NegativeTo


class PosNegNeuLabelsFormatter(StringLabelsFormatter):

    def __init__(self):
        stol = {
            "negative": PositiveTo,
            "positive": NegativeTo,
            "no-label": NoLabel
        }
        super(PosNegNeuLabelsFormatter, self).__init__(stol=stol)