import sys

from arekit_ss.core.utils.lexicons.rusentilex import RuSentiLexLexicon

sys.path.append('../../../../')

lexicon = RuSentiLexLexicon.from_zip()
for term in lexicon:
    print(term)

print('порядочный' in lexicon)
