'''
completion (language model) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, line-too-long
# SciPy is very chatty.
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pytest

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatAnthropic
from langchain.llms.openai import OpenAI, BaseOpenAI
from persyn.utils.color_logging import log

from src.persyn.interaction.completion import LanguageModel, setup_llm

# Bot config
from src.persyn.utils.config import load_config

@pytest.fixture(scope='module')
def lm() -> LanguageModel:
    return LanguageModel(config=load_config())

@pytest.mark.parametrize("input_keywords, expected_keywords", [
    (f'this thing, that thing, {{bot_name}}, the other thing', ['that thing', 'the other thing', 'this thing']),
    ('#codebase #newfeatures #hackerfriendly #conversation #thankful', ['codebase', 'conversation', 'hackerfriendly', 'newfeatures', 'thankful']),
    ('-Greeting', ['Greeting']),
    (' #Hackerfriendly #BugFix #Discord #DirectMessages', ['BugFix', 'DirectMessages', 'Discord', 'Hackerfriendly']),
    ('-AI', ['AI']),
    ('#Representation #Media #Acceptance #StockPhoto', ['Acceptance', 'Media', 'Representation', 'StockPhoto']),
    ('- Artificial Intelligence (AI)', ['Artificial Intelligence (AI)']),
    (r'#memory #art, memory, Art, \ #MEMORY, - art', ['Art', 'MEMORY', 'art', 'memory']),
    ('The Original Series', ['The Original Series']),
    ('#spock #paintings #manwithacigarette', ['manwithacigarette', 'paintings', 'spock']),
    ('memory, amnesia, scientist', ['amnesia', 'memory', 'scientist']),
    ('- Spock', ['Spock']),
])
def test_keywords(lm, input_keywords, expected_keywords):
    assert lm.cleanup_keywords(input_keywords.format(bot_name=lm.bot_name)) == expected_keywords

# Test the setup_llm function
def test_setup_llm():
    test_config = load_config()

    chat_llm = setup_llm(test_config, model='gpt-3.5-turbo')
    assert isinstance(chat_llm, ChatOpenAI)

    chat_anthropic = setup_llm(test_config, model='claude-2')
    if chat_anthropic is None:
        pytest.skip("Anthropic key not found in config, skipping Anthropic test.")
    assert isinstance(chat_anthropic, ChatAnthropic)

# Test the toklen method
def test_toklen(lm):
    text = "This is a test sentence."
    token_length = lm.toklen(text, model='gpt-3.5-turbo')
    assert isinstance(token_length, int)
    assert token_length == 6

# Test the paginate method
def test_paginate(lm):
    text = "This is a test sentence. " * 100  # Long text to ensure pagination
    max_tokens = 10  # Arbitrary small number for testing
    pages = list(lm.paginate(text, max_tokens=max_tokens))
    assert len(pages) == 60
    for page in pages:
        assert lm.toklen(page) <= max_tokens

# Test the trim method
def test_trim(lm):
    text = "This is a test sentence."
    trimmed_text = lm.trim(text)
    assert trimmed_text == text  # Nothing to trim

    text = "This is a test sentence. It also has a dangling word and"
    trimmed_text = lm.trim(text)
    assert trimmed_text.endswith("sentence.")  # The last incomplete sentence should be removed

    text = "This is a question? It also has a dangling word and"
    trimmed_text = lm.trim(text)
    assert trimmed_text.endswith("question?")  # The last incomplete sentence should be removed

    text = "This is important!!1! It also has a dangling word and"
    trimmed_text = lm.trim(text)
    assert trimmed_text.endswith("important!!1!")  # The last incomplete sentence should be removed

    text = "This is a test sentence with a dangling word and"
    trimmed_text = lm.trim(text)
    assert trimmed_text.endswith("and")  # Only trim if there is at least one complete sentence

# Test the truncate method
def test_truncate(lm):
    text = "This is a test sentence. " * 1000  # Long text to ensure truncation
    truncated_text = lm.truncate(text)
    assert len(truncated_text) <= len(text)  # Truncated text should be shorter or equal
    assert lm.toklen(truncated_text) <= lm.max_prompt_length()

# Test the get_embedding method
def test_get_embedding(lm):
    text = "This is a test sentence."
    embedding = lm.get_embedding(text)
    assert isinstance(embedding, bytes)
    assert len(embedding) == 6144  # 4 bytes per vector * 1536

# Test the summarize_text method
def test_summarize_text(lm):

    # Get the current log level
    text = """Cats, you know, they're like little bundles of grace with a side of sass. They've got these soft fur coats that you just can't help but wanna pet, and when they do that slow blink at you, it's like they're saying "I trust you" without making a big deal out of it. And let's not forget about their playfulness; give 'em a simple cardboard box and they turn it into the ultimate adventure playground. Plus, they've got this independent streak that's pretty admirable; they do their own thing but still come around for cuddles when it suits them. In short, cats are paws-down adorable and pretty darn awesome."""

    summary = lm.summarize_text(text)
    log.info(f'\n{summary}')
    assert isinstance(summary, str)
    assert len(summary) <= len(text)  # Summary should be shorter or equal

    haiku = lm.summarize_text(text, summarizer="Create a haiku of the following text. Your response must include only the haiku and no other text:")
    log.info(f'\n{haiku}')
    assert len(haiku) < len(summary)


# Test the cosine_similarity method
def test_cosine_similarity(lm):
    a = np.array([1, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0], dtype=np.float32)
    similarity = lm.cosine_similarity(a, b)
    assert similarity == 0  # Orthogonal vectors have a cosine similarity of 0

    similarity = lm.cosine_similarity(a, a)
    assert similarity == 1  # Identical vectors have a cosine similarity of 1

    c = np.array([1, 0.7, 0.5], dtype=np.float32)
    similarity = lm.cosine_similarity(b, c)
    assert similarity ==  pytest.approx(0.530669, rel=1e-6)  # Similar vectors have a cosine similarity between 0 and 1

    similarity = lm.cosine_similarity(a, c)
    assert similarity ==  pytest.approx(0.758098, rel=1e-6)  # Similar vectors have a cosine similarity between 0 and 1

# Test the cleanup_keywords method
def test_cleanup_keywords(lm):
    text = f"keyword1,- keyword2, 7, #keyword3  , * keyword4, {lm.bot_name}"
    keywords = lm.cleanup_keywords(text)
    assert isinstance(keywords, list)
    assert "keyword1" in keywords
    assert "keyword2" in keywords
    assert "keyword3" in keywords
    assert "keyword4" in keywords
    assert "7" not in keywords
    assert lm.bot_name.lower() not in keywords

# Test the camelCaseName static method
def test_camelCaseName():
    name = "test name"
    camel_case_name = LanguageModel.camelCaseName(name)
    assert camel_case_name == "testName"
    assert len(camel_case_name) <= 64

    longname = "this is a very long name that is longer than 64 characters and should be truncated"
    camel_case_name = LanguageModel.camelCaseName(longname)
    assert camel_case_name == "thisIsAVeryLongNameThatIsLongerThan64CharactersAndShouldBeTrunca"
    assert len(camel_case_name) == 64

# Test cases for the safe_name function
@pytest.mark.parametrize("input_name, expected_output", [
    ("ValidName123", "ValidName123"),  # Alphanumeric characters should remain unchanged
    ("Name with spaces", "Name with spaces"),  # Spaces should be allowed
    ("Name,with,commas,", "Name,with,commas,"),  # Commas should be allowed
    ("Special!@#$%^&*()Chars", "SpecialChars"),  # Special characters should be removed
    ("   Leading and trailing spaces   ", "Leading and trailing spaces"),  # Leading/trailing spaces should be stripped
    ("NameWithÜñíçøɗé", "NameWith"),  # Non-ASCII characters should be removed
    ("1234567890123456789012345678901234567890123456789012345678901234567890", "1234567890123456789012345678901234567890123456789012345678901234"),  # Length should be truncated to 64 characters
    ("", ""),  # Empty string should return empty
    ("Only,Commas,,,,", "Only,Commas,,,,"),
    ("Tabs\tand\nNewlines\n", "TabsandNewlines"),
    ("Mixed-_@.,# Spaces and, Commas", "Mixed, Spaces and, Commas"),
    ("1234567890" * 7, "1234567890" * 6 + "1234"),  # Test exact boundary condition
    ("   Spaces   on   both   ends ", "Spaces   on   both   ends"),  # Multiple spaces inside should remain
    ("Name with\nnewline", "Name withnewline"),  # Newlines should be removed
])

def test_safe_name(lm, input_name, expected_output):
    assert lm.safe_name(input_name) == expected_output

# Test that the output is never longer than 64 characters
def test_safe_name_length(lm):
    long_name = "a" * 100  # Create a string longer than 64 characters
    sanitized_name = lm.safe_name(long_name)
    assert len(sanitized_name) <= 64

# Test that the output does not contain any disallowed characters
def test_safe_name_characters(lm):
    name_with_everything = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789, !@#$%^&*()_+-=[]{}|;:'\"<>,.?/~`"
    sanitized_name = lm.safe_name(name_with_everything)
    for char in sanitized_name:
        assert char.isalnum() or char == ',' or char == ' '

# Test that leading and trailing whitespace is removed
def test_safe_name_whitespace(lm):
    name_with_whitespace = "   surrounded by spaces   "
    sanitized_name = lm.safe_name(name_with_whitespace)
    assert sanitized_name == "surrounded by spaces"
    assert sanitized_name[0] != " "
    assert sanitized_name[-1] != " "
