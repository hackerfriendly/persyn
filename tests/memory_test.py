'''
memory (redis) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, no-member
import uuid
import datetime as dt

import ulid
import pytest

from langchain.memory import (
    CombinedMemory,
    ConversationSummaryBufferMemory,
    ConversationKGMemory
)
from langchain.vectorstores.redis import Redis

from src.persyn.interaction.memory import Recall, Convo, escape, scquery

# Bot config
from src.persyn.utils.config import load_config

# from utils.color_logging import log

persyn_config = load_config()

def clear_ns(ns, chunk_size=5000):
    ''' Clear a namespace '''
    recall = Recall(persyn_config)

    if not ns or ':' not in ns:
        return False

    cursor = '0'
    while cursor != 0:
        cursor, keys = recall.redis.scan(cursor=cursor, match=f"{ns}*", count=chunk_size)
        if keys:
            recall.redis.delete(*keys)
    return True

@pytest.fixture
def cleanup():
    ''' Delete everything with the test bot_id '''
    yield

    recall = Recall(persyn_config)
    clear_ns(recall.convo_prefix)

    for idx in [recall.convo_prefix, recall.opinion_prefix, recall.goal_prefix, recall.news_prefix]:
        try:
            recall.redis.ft(idx).dropindex()
        except recall.redis.exceptions.ResponseError as err:
            print(f"Couldn't drop index {idx}:", err)

@pytest.fixture
def recall(conversation_interval=2):
    return Recall(persyn_config, conversation_interval)

def test_escape():
    assert escape("test!@#") == "test\\!\\@\\#"

# Test the scquery function
def test_scquery():
    assert scquery(service="test_service", channel="test_channel") == "(@service:{test_service}) (@channel:{test_channel})"
    assert scquery(service="test_service") == "(@service:{test_service})"
    assert scquery(channel="test_channel") == "(@channel:{test_channel})"
    assert scquery() == ""

# Test the Convo class
def test_convo():
    convo = Convo(service="test_service", channel="test_channel")
    assert convo.service == "test_service"
    assert convo.channel == "test_channel"
    assert isinstance(convo.id, str)

    assert repr(convo) == f"service='test_service', channel='test_channel', id='{convo.id}'"
    assert str(convo) == f"test_service|test_channel|{convo.id}"

    assert convo.visited == set([convo.id])

def test_recall_initialization():
    recall = Recall(persyn_config, conversation_interval=2)
    assert recall.bot_name == 'Test'
    assert str(recall.bot_id) == 'ffffffff-cafe-1337-feed-facade123456'

def test_fetch_summary(recall, cleanup):
    # Setup: create a conversation with a known summary
    convo_id = "test_convo_id"
    expected_summary = "This is a test summary."
    recall.redis.hset(f"{recall.convo_prefix}:{convo_id}:summary", "content", expected_summary)

    # Exercise: fetch the summary
    summary = recall.fetch_summary(convo_id)

    # Verify: the fetched summary matches the expected summary
    assert summary == expected_summary

def test_decode_dict(recall):
    assert recall.decode_dict({}) == {}

    input_dict = {b'key1': b'value1', b'key2': b'value2'}
    expected_dict = {'key1': 'value1', 'key2': 'value2'}
    assert recall.decode_dict(input_dict) == expected_dict

    input_dict = {'key1': 'value1', 'key2': 'value2'}
    with pytest.raises(AttributeError):
        recall.decode_dict(input_dict)

    input_dict = {b'key1': b'value1', b'key2': 123}
    with pytest.raises(AttributeError):
        recall.decode_dict(input_dict)

    input_dict = {123: b'value1', 456: b'value2'}
    with pytest.raises(AttributeError):
        recall.decode_dict(input_dict)

    input_dict = {b'key1': b'value1', b'key2': b'value2'}
    expected_dict = {'key1': 'value1', 'key2': 'value2'}
    decoded_dict = recall.decode_dict(input_dict)
    assert decoded_dict == expected_dict

def test_create_lc_memories(recall):
    memories = recall.create_lc_memories()

    # Verify dictionary with correct keys
    expected_keys = ["combined", "summary", "kg", "redis"]
    assert all(key in memories for key in expected_keys)

    # Verify correct types
    assert isinstance(memories["combined"], CombinedMemory)
    assert isinstance(memories["summary"], ConversationSummaryBufferMemory)
    assert isinstance(memories["kg"], ConversationKGMemory)
    assert isinstance(memories["redis"], Redis)

def test_set_and_fetch_convo_meta(recall, cleanup):
    convo_id = "test_convo_id"
    meta_key = "test_meta_key"
    meta_value = "test_meta_value"

    # Set metadata
    recall.set_convo_meta(convo_id, meta_key, meta_value)

    # Fetch metadata
    fetched_value = recall.fetch_convo_meta(convo_id, meta_key)

    assert fetched_value == meta_value

def test_list_convo_ids(recall, cleanup):
    # Setup: create multiple conversations
    service = "test_service_list_convo_ids"
    channel = "test_channel_list_convo_ids"
    speaker = "test_speaker"

    convo_ids = [convo.id for convo in [recall.new_convo(service, channel, speaker) for _ in range(3)]]

    assert convo_ids[0] != convo_ids[1] != convo_ids[2]
    assert list(recall.list_convo_ids(service, channel)) == convo_ids
    for convo_id, meta in recall.list_convo_ids(service, channel).items():
        assert meta['service'] == service
        assert meta['channel'] == channel

    assert list(recall.list_convo_ids(service, channel, expired=True)) == []
    assert list(recall.list_convo_ids(service, channel, expired=False)) == convo_ids

    # Exercise: expire one conversation
    recall.expire_convo(convo_ids[1])

    # Verify: expired conversation is not listed
    assert list(recall.list_convo_ids(service, channel, expired=False)) == [convo_ids[0], convo_ids[2]]
    assert list(recall.list_convo_ids(service, channel, expired=True)) == [convo_ids[1]]

    # Still listed when expired=None
    assert list(recall.list_convo_ids(service, channel)) == [convo_ids[0], convo_ids[1], convo_ids[2]]

    # Excersize: test paging. This returns the newest conversation first.
    assert list(recall.list_convo_ids(service, channel, expired=False, size=1)) == [convo_ids[2]]

    # Expire the other conversations
    recall.expire_convo(convo_ids[0])
    recall.expire_convo(convo_ids[2])

    # Verify: all conversations are expired
    assert list(recall.list_convo_ids(service, channel, expired=False)) == []
    assert list(recall.list_convo_ids(service, channel, expired=True)) == [convo_ids[0], convo_ids[1], convo_ids[2]]

    # Test after
    recall.set_convo_meta(convo_ids[0], "expired_at", dt.datetime.now().timestamp() - 1000)
    assert list(recall.list_convo_ids(service, channel, expired=True, after=10)) == [convo_ids[1], convo_ids[2]]
    assert list(recall.list_convo_ids(service, channel, expired=True, after=1001)) == [convo_ids[0], convo_ids[1], convo_ids[2]]

def test_get_last_convo_id(recall, cleanup):
    service = "test_service"
    channel = "test_channel"
    last_convo_id = "last_convo_id"

    # Setup: simulate conversation creation
    recall.set_convo_meta(last_convo_id, "service", service)
    recall.set_convo_meta(last_convo_id, "channel", channel)
    recall.set_convo_meta(last_convo_id, "initiator", "test")
    recall.set_convo_meta(last_convo_id, "expired", "False")

    # Exercise: get the last conversation ID
    fetched_id = recall.get_last_convo_id(service, channel)

    # Verify: fetched ID matches expected last_convo_id
    assert fetched_id == last_convo_id

    # Cleanup
    recall.redis.delete(f"{recall.convo_prefix}:{last_convo_id}:meta")


def test_get_last_message_id(recall, cleanup):
    convo_id = "test_convo_id"
    last_message_id = "last_message_id"
    final_message_id = "zzz"

    # Setup: simulate message creation
    recall.redis.set(f"{recall.convo_prefix}:{convo_id}:dialog:last_message_id", last_message_id)

    # Exercise: get the last message ID
    fetched_id = recall.get_last_message_id(convo_id)

    # Verify: fetched ID matches expected last_message_id
    assert fetched_id == last_message_id

    # Exercise: add more message IDs
    recall.redis.set(f"{recall.convo_prefix}:{convo_id}:dialog:zzz", final_message_id)
    recall.redis.set(f"{recall.convo_prefix}:{convo_id}:dialog:aaa", "aaa")

    # Verify: fetched ID matches expected last_message_id
    fetched_id = recall.get_last_message_id(convo_id)
    assert fetched_id == final_message_id

    # Cleanup
    recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:dialog:last_message_id")
    recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:dialog:aaa")
    recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:dialog:zzz")

def test_current_convo_id(recall, cleanup):
    ''' Revisit this if multiple conversations are allowed at once. '''
    service = "test_service"
    channel = "test_channel"
    speaker_name = "test_speaker"

    # Setup: start a conversation creation
    first_convo = recall.new_convo(service, channel, speaker_name)

    # Verify: current ID matches first_convo.id
    assert first_convo.id == recall.current_convo_id(service, channel)

    # Exercise: add another conversation ID
    second_convo = recall.new_convo(service, channel, speaker_name)

    # Verify: current ID matches second_convo.id
    assert first_convo.id != second_convo.id
    assert recall.current_convo_id(service, channel) == second_convo.id

    # Exercise: expire the second conversation
    recall.expire_convo(second_convo.id)

    # Verify: No current convo (one convo at a time!)
    assert recall.current_convo_id(service, channel) == None

    # Exercise: create another conversation
    third_convo = recall.new_convo(service, channel, speaker_name)

    # Verify: current ID matches third_convo.id
    assert third_convo.id == recall.current_convo_id(service, channel)

    # Expire the conversation
    recall.expire_convo(third_convo.id)

    # Verify: No current convo
    assert recall.current_convo_id(service, channel) == None

def test_post_recall_init(recall):
    assert recall.bot_name == recall.config.id.name
    assert recall.bot_id == uuid.UUID(recall.config.id.guid)
    assert recall.conversation_interval == recall.config.memory.conversation_interval
    assert recall.max_summary_size == recall.config.memory.max_summary_size
    # Add more assertions for the indices and Redis commands if necessary

def test_id_to_epoch():
    # Generate a ULID and test conversion
    ulid_str = str(ulid.ULID())
    epoch = Recall.id_to_epoch(ulid_str)
    assert epoch > 0

# Test id_to_timestamp method
def test_id_to_timestamp(recall):
    # Generate a ULID and test conversion
    ulid_str = str(ulid.ULID())
    timestamp = recall.id_to_timestamp(ulid_str)
    assert type(timestamp) == str

    # Test with a ULID from a known timestamp
    the_epoch = 1705113872.0
    the_timestamp = "2024-01-12T18:44:32-08:00"

    ulid_str = str(ulid.ULID().from_timestamp(the_epoch))

    epoch = recall.id_to_epoch(ulid_str)
    assert type(epoch) == float
    assert epoch == the_epoch

    timestamp = recall.id_to_timestamp(ulid_str)
    assert type(timestamp) == str
    assert timestamp == the_timestamp

def test_new_convo_without_convo_id(recall, cleanup):
    service = "test_service"
    channel = "test_channel"
    speaker_name = "test_speaker"

    # Start a new conversation without a convo_id
    convo = recall.new_convo(service, channel, speaker_name)

    # Verify that a Convo instance is returned
    assert isinstance(convo, Convo)

    # Verify the Convo instance has the correct attributes
    assert convo.service == service
    assert convo.channel == channel
    assert convo.id is not None  # A convo_id should be generated

    # Verify that conversation metadata is set correctly in Redis
    assert recall.fetch_convo_meta(convo.id, "service") == service
    assert recall.fetch_convo_meta(convo.id, "channel") == channel
    assert recall.fetch_convo_meta(convo.id, "initiator") == speaker_name
    assert recall.fetch_convo_meta(convo.id, "expired") == "False"

    # Clean up after the test
    recall.redis.delete(convo.id)

# Test convo_expired method
def test_convo_expired(recall, cleanup):
    # Test with no convo_id and no service and channel
    service = "test_service"
    channel = "test_channel"
    speaker_name = "test_speaker"

    with pytest.raises(RuntimeError):
        recall.convo_expired()

    # Start a new conversation without a convo_id
    convo = recall.new_convo(service, channel, speaker_name)
    assert recall.convo_expired(service, channel, convo.id) is False

    # Test manual expiration
    recall.set_convo_meta(convo.id, "expired", "True")
    assert recall.convo_expired(service, channel, convo.id) is True

    # Test automatic expiration
    recall.set_convo_meta(convo.id, "expired", "False")
    assert recall.convo_expired(service, channel, convo.id) is False

    # Backdate the conversation
    line_id = str(ulid.ULID().from_datetime(dt.datetime(2021, 1, 1)))
    recall.redis.hset(f"{recall.convo_prefix}:{convo.id}:dialog:{line_id}", "service", service)
    recall.redis.hset(f"{recall.convo_prefix}:{convo.id}:dialog:{line_id}", "channel", channel)

    # Convo should now be expired
    assert recall.convo_expired(service, channel, convo.id) is True

def test_convo_expired_with_convo_id(recall, cleanup):
    # Test with a convo_id
    service = "test_service"
    channel = "test_channel"
    speaker_name = "test_speaker"

    # Backdate the conversation
    convo_id = str(ulid.ULID().from_datetime(dt.datetime(2021, 1, 1)))

    # Start a new conversation without an expired convo_id and no conversation
    convo = recall.new_convo(service, channel, speaker_name, convo_id)
    assert recall.convo_expired(service, channel, convo.id) is True

    # Revive the convo
    new_convo_id = str(ulid.ULID())
    recall.redis.hset(f"{recall.convo_prefix}:{convo.id}:dialog:{new_convo_id}", "service", service)
    recall.set_convo_meta(convo.id, "expired", "False")
    assert recall.convo_expired(service, channel, convo.id) is False

    # Now expire it again
    recall.redis.delete(f"{recall.convo_prefix}:{convo.id}:dialog:{new_convo_id}")
    recall.redis.hset(f"{recall.convo_prefix}:{convo.id}:dialog:{convo_id}", "service", service)
    recall.redis.hset(f"{recall.convo_prefix}:{convo.id}:dialog:{convo_id}", "channel", channel)

    # Convo should now be expired
    assert recall.convo_expired(service, channel, convo.id) is True

def test_post_convo_init():
    convo = Convo(service="test_service", channel="test_channel")
    assert isinstance(convo.id, str)
    assert len(convo.id) == 26  # ULID should be 26 characters long

    convo_with_id = Convo(service="test_service", channel="test_channel", id="01F3ZDZXP3JZBZ30D8X6R0A2P2")
    assert convo_with_id.id == "01F3ZDZXP3JZBZ30D8X6R0A2P2"

def test_memories_initialization():
    convo = Convo(service="test_service", channel="test_channel")
    assert isinstance(convo.memories, dict)
    assert len(convo.memories) == 0

def test_visited_initialization():
    convo = Convo(service="test_service", channel="test_channel")
    assert isinstance(convo.visited, set)
    assert len(convo.visited) == 1
    assert convo.id in convo.visited

def test_repr():
    convo = Convo(service="test_service", channel="test_channel")
    assert repr(convo) == f"service='test_service', channel='test_channel', id='{convo.id}'"

def test_str():
    convo = Convo(service="test_service", channel="test_channel")
    assert str(convo) == f"test_service|test_channel|{convo.id}"

def test_find_related_convos(recall, cleanup):
    service = 'test_service'
    channel = 'test_channel'
    human_name = 'human'
    bot_name = 'bot'

    # Create a conversation
    convo = recall.new_convo(service, channel, human_name)

    human = {
        "service": service,
        "channel": channel,
        "convo_id": convo.id,
        "speaker_name": human_name,
        "verb": "summary"
    }

    bot = {
            "service": service,
            "channel": channel,
            "convo_id": convo.id,
            "speaker_name": bot_name,
            "verb": "summary"
    }

    convo.memories['redis'].add_texts(
        [
            "What's new?",
            "Not much, what's new with you?",
            "Do you think raccoons can dance?",
            "Why are you so weird?"
        ],
        metadatas=[
            human,
            bot,
            human,
            bot
        ],
        keys=[
            f'{convo.id}:dialog:{str(ulid.ULID())}',
            f'{convo.id}:dialog:{str(ulid.ULID())}',
            f'{convo.id}:dialog:{str(ulid.ULID())}',
            f'{convo.id}:dialog:{str(ulid.ULID())}'
        ]
    )

    # Test exact match
    related = recall.find_related_convos(service, channel, "Do you think raccoons can dance?", threshold=1, size=1)
    assert len(related) == 1
    assert related[0][0] == convo.id
    assert related[0][1] < 0.0001

    # Test approximate match
    related = recall.find_related_convos(service, channel, "Can animals move and wiggle?", threshold=1, size=10)
    assert len(related) == 4
    assert related[0][0] == convo.id
    assert related[0][1] < 0.2
    assert related[1][1] > 0.2

    # Test no match for strings < 20 characters
    assert len(recall.find_related_convos(service, channel, "", threshold=1, size=10)) == 0
    assert len(recall.find_related_convos(service, channel, "aaa", threshold=1, size=10)) == 0
    assert len(recall.find_related_convos(service, channel, "aaaa", threshold=1, size=10)) == 4

    # Threshold cutoff
    assert len(recall.find_related_convos(service, channel, "Can animals move and wiggle?", threshold=0.2, size=10)) == 1

def test_add_goal(recall, cleanup):
    goal_id = recall.add_goal("service", "channel", "goal")
    assert goal_id is not None

def test_add_goal_action(recall, cleanup):
    goal_id = recall.add_goal("service", "channel", "goal")
    recall.add_goal_action(goal_id, "action")
    actions = recall.list_goal_actions(goal_id)
    assert "action" in actions
    assert len(actions) == 1

    recall.add_goal_action(goal_id, "action")
    actions = recall.list_goal_actions(goal_id)
    assert len(actions) == 1 # duplicate actions not permitted

    recall.add_goal_action(goal_id, "action2")
    actions = recall.list_goal_actions(goal_id)
    assert len(actions) == 2

def test_delete_goal_action(recall, cleanup):
    goal_id = recall.add_goal("service", "channel", "goal")
    recall.add_goal_action(goal_id, "action")
    recall.delete_goal_action(goal_id, "action")
    actions = recall.list_goal_actions(goal_id)
    assert "action" not in actions

def test_list_goal_actions(recall, cleanup):
    goal_id = recall.add_goal("service", "channel", "goal")
    recall.add_goal_action(goal_id, "action")
    actions = recall.list_goal_actions(goal_id)
    assert actions == ["action"]

def test_fetch_goal(recall, cleanup):
    goal_id = recall.add_goal("service", "channel", "goal")
    goal = recall.fetch_goal(goal_id)
    assert goal["content"] == "goal"
    assert goal["goal_id"] == goal_id

def test_list_goals_for_channel(recall, cleanup):
    recall.add_goal("service", "channel", "goal")
    goals = recall.list_goals_for_channel("service", "channel")
    assert len(goals) == 1

    recall.add_goal("service", "channel", "goal2")
    goals = recall.list_goals_for_channel("service", "channel")
    assert len(goals) == 1 # default size == 1

    goals = recall.list_goals_for_channel("service", "channel", size=10)
    assert len(goals) == 2

    recall.add_goal("service", "channel", "goal2")
    goals = recall.list_goals_for_channel("service", "channel", size=10)
    assert len(goals) == 3 # duplicate goals permitted

def test_achieve_goal(recall, cleanup):
    goal_id = recall.add_goal("service", "channel", "goal")
    assert recall.goal_achieved(goal_id) == False

    recall.achieve_goal(goal_id, 0.9)
    assert recall.goal_achieved(goal_id) == False

    recall.achieve_goal(goal_id, 0.1)
    assert recall.goal_achieved(goal_id) == True

    recall.achieve_goal(goal_id, -0.1)
    assert recall.goal_achieved(goal_id) == False
