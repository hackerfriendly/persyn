import pytest
import time
import datetime as dt

from unittest.mock import Mock
from regex import F

import ulid
from persyn.interaction.memory import Convo
from persyn.utils.color_logging import log

from src.persyn.interaction.interact import Interact
from src.persyn.interaction.chrono import hence

# Bot config
from src.persyn.utils.config import load_config

persyn_config = load_config()

@pytest.fixture
def mock_language_model():
    mock = Mock()
    # Configure the mock to return a fake response when prompted
    mock.chat_llm.invoke.return_value = {'response': 'Mocked response'}
    mock.trim.return_value = 'Mocked response'
    return mock

def clear_ns(ns, chunk_size=5000):
    ''' Clear a namespace '''
    interact = Interact(persyn_config)

    if not ns or ':' not in ns:
        return False

    cursor = '0'
    while cursor != 0:
        cursor, keys = interact.recall.redis.scan(cursor=cursor, match=f"{ns}*", count=chunk_size)
        if keys:
            interact.recall.redis.delete(*keys)
    return True

@pytest.fixture
def cleanup():
    ''' Delete everything with the test bot_id '''
    yield

    interact = Interact(persyn_config)
    clear_ns(interact.recall.convo_prefix)

    for idx in [interact.recall.convo_prefix, interact.recall.opinion_prefix, interact.recall.goal_prefix, interact.recall.news_prefix]:
        try:
            interact.recall.redis.ft(idx).dropindex()
        except interact.recall.redis.exceptions.ResponseError as err:
            print(f"Couldn't drop index {idx}:", err)

@pytest.fixture
def interact() -> Interact:
    return Interact(persyn_config)


def test_template(interact: Interact):
    # Test that the template method returns the expected string format
    context = "Test context"
    result = interact.template(context)
    assert context in result
    assert result.startswith("It is ")
    for key in ["kg", "history", "human", "bot", "input"]:
        assert f"{{{key}}}" in result

    # Test with embedded {}
    context = "Test {context} with {embedded} {{brackets}} and (parentheses)"
    stripped = context.replace("{", "(").replace("}", ")")
    result = interact.template(context)
    assert stripped in result

@pytest.mark.parametrize("seconds_ago, expected", [
    (0, ""),  # Recent timestamp
    (7199, ""),  # Almost recent timestamp
    (7201, "a conversation from 2 hours ago:"),  # Greater than 7200 seconds ago
    (7 * 24 * 3600, "a conversation from 7 days ago:"),
    (45 * 24 * 3600, "a conversation from a month ago:"),
    (365 * 50 * 24 * 3600, "a conversation from 50 years ago:")
])
def test_get_time_preamble(interact, seconds_ago, expected):
    the_timestamp = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds_ago))
    ulid_str = str(ulid.ULID().from_datetime(the_timestamp))
    result = interact.get_time_preamble(ulid_str)
    assert result.strip() == expected

def test_too_many_tokens(interact):
    # Test that too_many_tokens returns True when the input is too long
    convo = interact.recall.new_convo('service', 'channel', 'Rob')
    max_tokens = int(interact.lm.max_prompt_length() * persyn_config.memory.context)

    assert interact.too_many_tokens(convo, "", 0) == False
    assert interact.too_many_tokens(convo, "", max_tokens) == False
    assert interact.too_many_tokens(convo, "", max_tokens + 1) == True

    assert interact.too_many_tokens(convo, "cat " * max_tokens, 0) == False
    assert interact.too_many_tokens(convo, "cat " * (max_tokens + 1), 0) == True

    convo.memories['summary'].chat_memory.add_ai_message("cat " * (max_tokens - 4))
    log.warning(convo.memories['summary'].load_memory_variables({})['history'])

    assert interact.too_many_tokens(convo, "", 0) == False
    assert interact.too_many_tokens(convo, "", 1) == False
    assert interact.too_many_tokens(convo, "cat", 0) == False
    assert interact.too_many_tokens(convo, "cat", 1) == True

def test_get_sentiment_analysis(interact):
    convo = interact.recall.new_convo('service', 'channel', 'Rob')

    assert interact.get_sentiment_analysis(convo) == ("sentiment analysis", f"{persyn_config.id.name} is feeling nothing in particular.")

    interact.recall.set_convo_meta(convo.id, "feels", "delighted and excited to be part of this test")
    assert interact.get_sentiment_analysis(convo) == ("sentiment analysis", f"{persyn_config.id.name} is feeling delighted and excited to be part of this test.")


@pytest.fixture
def setup_conversations(interact, cleanup):
    # Create and expire several conversations to generate summaries
    service = "summary_service"
    channel = "summary_channel"
    speaker_name = "test_speaker"

    for i in range(5):
        # Create a convo
        convo = interact.recall.new_convo(service, channel, speaker_name)
        # Add a summary
        convo.memories['redis'].add_texts(
            texts=[f"summary {i}"],
            metadatas=[
                {
                    "service": convo.service,
                    "channel": convo.channel,
                    "convo_id": convo.id,
                    "speaker_name": "narrator",
                    "verb": "summary",
                    "role": "bot"
                },
            ],
            keys=[f"{convo.id}:summary"]
        )
        # Expire the convo
        interact.recall.expire_convo(convo.id)


def test_get_recent_summaries(interact, setup_conversations):
    service = "summary_service"
    channel = "summary_channel"
    speaker_name = "test_speaker"

    convo = interact.recall.new_convo(service, channel, speaker_name)

    summaries = interact.get_recent_summaries(convo)

    # Check that we have summaries and they are in the correct order (most recent first)
    assert summaries
    assert all("recent summary" in summary[0] for summary in summaries)
    assert len(summaries) == 5  # Assuming we have 5 convos from setup_conversations
    # Check that the summaries are in the correct order
    for i in range(4):
        assert summaries[i][1] < summaries[i+1][1]


# def test_get_recent_summaries_with_limit(test_convo, setup_conversations):
#     your_class_instance = YourClass()
#     # Assuming that too_many_tokens will return True after a certain threshold
#     summaries = your_class_instance.get_recent_summaries(test_convo, used=100)

#     # Check that the number of summaries is limited due to token count
#     assert len(summaries) < 5


# def test_add_context(interact: Interact):

#     # Test that add_context appends the correct context information
#     # This may require mocking the Recall object and its methods

# def test_get_time_preamble(interact: Interact):
#     # Test that get_time_preamble returns the correct preamble based on time elapsed
#     # This will require mocking chrono.elapsed

# def test_current_dialog(interact):
#     # Test that current_dialog returns the correct dialog from the convo
#     # This will require setting up a mock Convo object with expected history

# def test_retort(interact, mock_language_model):
#     # Test that retort returns a trimmed response and sends a chat if send_chat is True
#     # This will require mocking out send_chat and checking that it was called with the correct parameters

# def test_status(interact):
#     # Test that status returns the correct prompt and chat history for a channel
#     # This will require setting up a mock Convo object and checking the returned prompt format

# def test_inject_idea(interact):
#     # Test that inject_idea correctly injects an idea into recall memory
#     # This will require mocking out the Recall object and its methods

# def test_summarize_channel(interact, mock_language_model):
#     # Test that summarize_channel returns a summary for a given channel
#     # This will require setting up a mock Convo object and mocking the summarize_text method

