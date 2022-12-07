from collections import OrderedDict

from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.contrib.source.sentinerel.labels import PositiveTo, NegativeTo


class PosNegNeuRelationsLabelScaler(BaseLabelScaler):

    def __init__(self):

        self.__int_to_label_dict = OrderedDict([
            (NoLabel(), 0),
            (PositiveTo(), 1),
            (NegativeTo(), -1),
        ])

        self.__uint_to_label_dict = OrderedDict([
            (NoLabel(), 0),
            (PositiveTo(), 1),
            (NegativeTo(), 2),
        ])

        super(PosNegNeuRelationsLabelScaler, self).__init__(int_dict=self.__int_to_label_dict,
                                                            uint_dict=self.__uint_to_label_dict)


class ThreeLabelScaler(SentimentLabelScaler):
    """ For frames annotation
    """

    def __init__(self):

        uint_labels = [(NoLabel(), 0),
                       (PositiveTo(), 1),
                       (NegativeTo(), 2)]

        int_labels = [(NoLabel(), 0),
                      (PositiveTo(), 1),
                      (NegativeTo(), -1)]

        super(ThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                               int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)
