'''
completion (language model) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, line-too-long
# SciPy is very chatty.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.persyn.interaction.completion import LanguageModel

# Bot config
from src.persyn.utils.config import load_config

completion = LanguageModel(config=load_config())

test_cases = {
    f'this thing, that thing, {completion.bot_name}, the other thing': ['that thing', 'the other thing', 'this thing'],
    '#codebase #newfeatures #hackerfriendly #conversation #thankful': ['codebase', 'conversation', 'hackerfriendly', 'newfeatures', 'thankful'],
    '-Greeting': ['greeting'],
    ' #Hackerfriendly #BugFix #Discord #DirectMessages': ['bugfix', 'directmessages', 'discord', 'hackerfriendly'],
    '-AI': ['ai'],
    '#Representation #Media #Acceptance #StockPhoto': ['acceptance', 'media', 'representation', 'stockphoto'],
    '- Artificial Intelligence (AI)': ['artificial intelligence (ai)'],
    r'#memory #art, memory, Art, \ #MEMORY, - art': ['art', 'memory'],
    'The Original Series': ['the original series'],
    '#spock #paintings #manwithacigarette': ['manwithacigarette', 'paintings', 'spock'],
    'memory, amnesia, scientist': ['amnesia', 'memory', 'scientist'],
    '- Spock': ['spock'],
}

def test_keywords():
    '''
    Test various (messy) keyword parsing heuristics.
    This test does not actually use the language model.
    '''
    for k, v in test_cases.items():
        assert completion.cleanup_keywords(k) == v
