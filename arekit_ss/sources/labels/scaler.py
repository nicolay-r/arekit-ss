from collections import OrderedDict

from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler

from arekit_ss.sources.labels.sentiment import NegativeTo, PositiveTo


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



