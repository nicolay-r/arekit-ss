import sys
import unittest

from arekit_ss.sources.rusentiframes.collection import RuSentiFramesCollection
from arekit_ss.sources.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, RuSentiFramesEffectLabelsFormatter
from arekit_ss.sources.rusentiframes.types import RuSentiFramesVersions

sys.path.append('../../')

from arekit.common.frames.variants.collection import FrameVariantsCollection

from labels import PositiveLabel, NegativeLabel


class TestRuSentiFrameVariants(unittest.TestCase):

    @staticmethod
    def __iter_frame_variants():
        frames_collection = RuSentiFramesCollection.read(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(
                neg_label_type=NegativeLabel, pos_label_type=PositiveLabel),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
                neg_label_type=NegativeLabel, pos_label_type=PositiveLabel))

        frame_variants = FrameVariantsCollection()
        frame_variants.fill_from_iterable(variants_with_id=frames_collection.iter_frame_id_and_variants(),
                                          overwrite_existed_variant=True,
                                          raise_error_on_existed_variant=False)

        for v, _ in frame_variants.iter_variants():
            yield v

    def test_iter_frame_variants(self):
        frame_values_list = list(self.__iter_frame_variants())
        for frame_variant in frame_values_list:
            print('"{}"'.format(frame_variant))


if __name__ == '__main__':
    unittest.main()
