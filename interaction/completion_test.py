'''
completion (language model) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, line-too-long
import sys
from pathlib import Path

# SciPy is very chatty.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from completion import LanguageModel

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../').resolve()))

# Bot config
from utils.config import load_config

completion = LanguageModel(config=load_config())

test_cases = {
    f'this thing, that thing, {completion.bot_name}, the other thing': ['other', 'that', 'the', 'thing', 'this'],
    '#codebase #newfeatures #hackerfriendly #conversation #thankful': ['codebase', 'conversation', 'hackerfriendly', 'newfeatures', 'thankful'],
    '#Friendship {Coding (Help) #Features': ['coding', 'features', 'friendship', 'help'],
    '-Greeting': ['greeting'],
    ' #Hackerfriendly #BugFix #Discord #DirectMessages': ['bugfix', 'directmessages', 'discord', 'hackerfriendly'],
    '-AI': ['ai'],
    '#Representation #Media #Acceptance #StockPhoto': ['acceptance', 'media', 'representation', 'stockphoto'],
    '- Artificial Intelligence (AI)': ['ai', 'artificial', 'intelligence'],
    r'#memory #art memory / Art \ MEMORY -art': ['art', 'memory'],
    'The Original Series': ['original', 'series', 'the'],
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
        assert completion.model.cleanup_keywords(k) == v
