import itertools
import unittest

from tqdm import tqdm

from arekit_ss.core.source.brat.doc import BratDocument
from arekit_ss.core.source.brat.relation import BratRelation
from arekit_ss.core.source.brat.sentence import BratSentence
from arekit_ss.sources.nerel_bio.utils.io_utils import NerelBioIOUtils
from arekit_ss.sources.nerel_bio.utils.reader import NerelBioDocReader
from arekit_ss.sources.nerel_bio.utils.versions import DEFAULT_VERSION


class TestNerelBioRead(unittest.TestCase):

    def test(self):
        doc_reader = NerelBioDocReader(version=DEFAULT_VERSION)
        news = doc_reader.read_document(filename="25558104_ru", doc_id=0)
        assert(isinstance(news, BratDocument))
        print("Sentences Count:", news.SentencesCount)
        for sentence in news.iter_sentences():
            assert(isinstance(sentence, BratSentence))
            print(sentence.Text.strip())
            for entity, bound in sentence.iter_entity_with_local_bounds():
                print("{}: ['{}',{}, {}]".format(
                    entity.ID, entity.Value, entity.Type,
                    "-".join([str(bound.Position), str(bound.Position+bound.Length)])))

        for brat_relation in news.Relations:
            assert(isinstance(brat_relation, BratRelation))
            print(brat_relation.SourceID, brat_relation.TargetID, brat_relation.Type)

    def test_all_documents(self, docs_limit=5):
        doc_reader = NerelBioDocReader(version=DEFAULT_VERSION)
        filenames_by_ids, folding = NerelBioIOUtils.read_dataset_split(version=DEFAULT_VERSION, docs_limit=docs_limit)
        for doc_id in tqdm(itertools.chain.from_iterable(folding.values())):
            doc_reader.read_document(filename=filenames_by_ids[doc_id], doc_id=0)
