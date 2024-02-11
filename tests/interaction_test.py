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

    assert interact.get_sentiment_analysis(convo) == ("sentiment analysis", f"{persyn_config.id.name}'s emotional state: neutral.", 7)

    interact.recall.set_convo_meta(convo.id, "feels", "delighted and excited to be part of this test")
    assert interact.get_sentiment_analysis(convo) == ("sentiment analysis", f"{persyn_config.id.name}'s emotional state: delighted and excited to be part of this test.", 15)


@pytest.fixture
def setup_conversations(interact, cleanup):
    # Create and expire several conversations to generate summaries
    service = "service"
    channel = "channel"
    speaker_name = "test_speaker"

    for i in range(5):
        # Create a convo
        convo = interact.recall.new_convo(service, channel, speaker_name)
        # Add a summary
        if i < 4:
            summary = f"A nice summary of Italian cooking {i}"
        else:
            # Last one is more specific, for relevance tests
            summary = f"The completely true story about aardvarks {i}"

        convo.memories['redis'].add_texts(
            texts=[summary],
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
        # Fake a final summary
        interact.recall.redis.hset(f"{interact.recall.convo_prefix}:{convo.id}:summary", "final", summary)

def test_get_recent_summaries(interact, setup_conversations):
    service = "service"
    channel = "channel"
    speaker_name = "test_speaker"

    convo = interact.recall.new_convo(service, channel, speaker_name)

    summaries = interact.get_recent_summaries(convo)

    # Check that we have summaries and they are in the correct order (most recent first)
    assert summaries
    assert all("recent summary" in summary[0] for summary in summaries)
    assert len(summaries) == 5  # We created 5 summaries in setup_conversations
    # Check that the summaries are in the correct order
    for i in range(4):
        assert summaries[i][1] < summaries[i+1][1]

def test_get_relevant_memories(interact, setup_conversations):
    service = "service"
    channel = "channel"
    speaker_name = "test_speaker"

    convo = interact.recall.new_convo(service, channel, speaker_name)
    assert len(convo.visited) == 1
    assert convo.id in convo.visited

    # Find the relevant memories from a fake conversation
    memories = interact.get_relevant_memories(convo, context=[('dialog', 'I love southern European food.')])
    assert len(memories) == 5

    # Query again should exclude those five, since they are now in visited
    memories = interact.get_relevant_memories(convo, context=[('dialog', 'Test recalls\nThe completely true story about aardvarks 4')])
    assert len(memories) == 0
    assert len(convo.visited) == 6
    # TODO: improve these tests to exercise relevance and expiration

def test_get_aardvark_memories(interact, setup_conversations):
    service = "service"
    channel = "channel"
    speaker_name = "test_speaker"

    convo = interact.recall.new_convo(service, channel, speaker_name)
    assert len(convo.visited) == 1
    assert convo.id in convo.visited

    # Find the relevant memories from a fake conversation
    memories = interact.get_relevant_memories(convo, context=[('dialog', 'Tell me that tale about aardvarks.')])
    assert len(memories) == 5

    # Query again should exclude those five, since they are now in visited
    memories = interact.get_relevant_memories(convo, context=[('dialog', 'Test recalls\nThe completely true story about aardvarks 4')])
    assert len(memories) == 0
    assert len(convo.visited) == 6


def test_inject_idea(interact, cleanup):
    # Prepare the test
    service = "test_service_inject_idea"
    channel = "test_channel_inject_idea"
    idea = "This is a test idea."
    verb = "recalls"

    # Convo must exist
    assert interact.inject_idea(service, channel, idea, verb) == False

    # Dialog not allowed
    assert interact.inject_idea(service, channel, idea, verb='dialog') == False

    # Create a new conversation
    convo = interact.recall.new_convo(service, channel, "test_speaker")

    # Inject the idea
    assert interact.inject_idea(service, channel, idea, verb) == True

    # Verify that the idea is saved in the summary
    convo = interact.recall.fetch_convo(service, channel)
    assert convo is not None
    summary = interact.recall.fetch_summary(convo.id)
    assert idea in summary

# TODO: Test retort, status, etc...

