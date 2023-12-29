from arekit_ss.sources.nerel.utils.reader import NerelDocReader
from arekit_ss.sources.nerel_bio.utils.io_utils import NerelBioIOUtils


class NerelBioDocReader(NerelDocReader):

    def __init__(self, version):
        super(NerelBioDocReader, self).__init__(version=version, io_utils=NerelBioIOUtils())
