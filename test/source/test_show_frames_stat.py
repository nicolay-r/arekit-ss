import sys
import unittest

sys.path.append('../../../../')

from test_rusentiframes_stat import about_version
from arekit_ss.sources.rusentiframes.types import RuSentiFramesVersions


class TestFramesStat(unittest.TestCase):

    def test_stat(self):
        about_version(version=RuSentiFramesVersions.V20)


if __name__ == '__main__':
    unittest.main()
