from arekit.common.data.doc_provider import DocumentProvider

from arekit_ss.sources.sentinerel.utils.io_utils import SentiNerelVersions
from arekit_ss.sources.sentinerel.utils.reader import SentiNerelDocReader


class SentiNERELDocProvider(DocumentProvider):
    """ Document reader for the collection of the RuSentNE competition 2023.
        For more details please follow the following repository:
        github: https://github.com/dialogue-evaluation/RuSentNE-evaluation
    """

    def __init__(self, filename_by_id, version):
        """ filename_ids: dict
                Dictionary of {id: filename}, where
                    - id: int
                    - filename: str
            version: SentiNerelVersions
                Specify the appropriate version of teh SentiNEREL collection.
        """
        assert(isinstance(filename_by_id, dict))
        assert(isinstance(version, SentiNerelVersions))
        super(SentiNERELDocProvider, self).__init__()
        self.__filename_by_id = filename_by_id
        self.__version = version

    def by_id(self, doc_id):
        return SentiNerelDocReader.read_document(doc_id=doc_id,
                                                 version=self.__version,
                                                 filename=self.__filename_by_id[doc_id])
