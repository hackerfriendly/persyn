'''
memory (redis) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, no-member
from unittest.mock import MagicMock, patch
import uuid

from time import sleep

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


@pytest.fixture(scope="module")
def recall(conversation_interval=2):
    return Recall(persyn_config, conversation_interval)

def test_fetch_summary(recall):
    # Setup: create a conversation with a known summary
    convo_id = "test_convo_id"
    expected_summary = "This is a test summary."
    recall.redis.hset(f"{recall.convo_prefix}:{convo_id}:summary", "content", expected_summary)

    # Exercise: fetch the summary
    summary = recall.fetch_summary(convo_id)

    # Verify: the fetched summary matches the expected summary
    assert summary == expected_summary

    # Cleanup: remove the test data from Redis
    recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:summary")

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

def test_set_and_fetch_convo_meta(recall):
    convo_id = "test_convo_id"
    meta_key = "test_meta_key"
    meta_value = "test_meta_value"

    # Set metadata
    recall.set_convo_meta(convo_id, meta_key, meta_value)

    # Fetch metadata
    fetched_value = recall.fetch_convo_meta(convo_id, meta_key)

    assert fetched_value == meta_value

    # Cleanup
    recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:meta")

def test_list_convo_ids(recall):
    # Setup: create multiple conversations
    convo_ids = ["convo1", "convo2", "convo3"]
    for convo_id in convo_ids:
        recall.set_convo_meta(convo_id, "expired", "False")

    recall.set_convo_meta("convo1", "expired", "True")

    # Exercise: list active conversation IDs
    active_ids = recall.list_convo_ids(active_only=True)

    # Verify: active_ids contains expected IDs
    assert set(active_ids).issubset(set(convo_ids))

    # Exercise: list all conversation IDs
    all_ids = recall.list_convo_ids(active_only=False)

    # Verify: all_ids contains expected IDs
    assert set(all_ids) == set(convo_ids)

    # Cleanup
    for convo_id in convo_ids:
        recall.redis.hdel(f"{recall.convo_prefix}:{convo_id}:meta", "expired")

def test_get_last_convo_id(recall):
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


def test_get_last_message_id(recall):
    convo_id = "test_convo_id"
    last_message_id = "last_message_id"
    final_message_id = "zzz"

    # Setup: simulate message creation
    recall.redis.set(f"{recall.convo_prefix}:{convo_id}:lines:last_message_id", last_message_id)

    # Exercise: get the last message ID
    fetched_id = recall.get_last_message_id(convo_id)

    # Verify: fetched ID matches expected last_message_id
    assert fetched_id == last_message_id

    # Exercise: add more message IDs
    recall.redis.set(f"{recall.convo_prefix}:{convo_id}:lines:zzz", final_message_id)
    recall.redis.set(f"{recall.convo_prefix}:{convo_id}:lines:aaa", "aaa")

    # Verify: fetched ID matches expected last_message_id
    fetched_id = recall.get_last_message_id(convo_id)
    assert fetched_id == final_message_id

    # Cleanup
    recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:lines:last_message_id")
    recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:lines:aaa")
    recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:lines:zzz")

def test_current_convo_id(recall):
    service = "test_service"
    channel = "test_channel"
    expected_convo_id = str(ulid.ULID())

    # Setup: simulate conversation creation
    recall.set_convo_meta(expected_convo_id, "service", service)
    recall.set_convo_meta(expected_convo_id, "channel", channel)
    recall.set_convo_meta(expected_convo_id, "initiator", "test")
    recall.set_convo_meta(expected_convo_id, "expired", "False")

    # Exercise: get the current conversation ID
    current_id = recall.current_convo_id(service, channel)

    # Verify: current ID matches expected_convo_id
    assert current_id == expected_convo_id

    # Exercise: add another conversation ID
    recall.set_convo_meta(expected_convo_id, "expired", "True")

    another_convo_id = str(ulid.ULID())

    recall.set_convo_meta(another_convo_id, "service", service)
    recall.set_convo_meta(another_convo_id, "channel", channel)
    recall.set_convo_meta(another_convo_id, "initiator", "test")
    recall.set_convo_meta(another_convo_id, "expired", "False")

    # Exercise: get the current conversation ID
    current_id = recall.current_convo_id(service, channel)

    # Verify: current ID matches expected_convo_id
    assert current_id == another_convo_id

    # Cleanup
    recall.redis.delete(f"{recall.convo_prefix}:{expected_convo_id}:meta")
    recall.redis.delete(f"{recall.convo_prefix}:{another_convo_id}:meta")

# ### 13. Test `expired`

# ```python
# def test_expired(recall):
#     id_to_test = "test_id"

#     # Setup: simulate an expired ID
#     recall.redis.setex(f"{recall.convo_prefix}:{id_to_test}", 1, "value")
#     time.sleep(2)  # Wait for the key to expire

#     # Exercise: check if the ID is expired
#     is_expired = recall.expired(id_to_test)

#     # Verify: ID is expired
#     assert is_expired
# ```

# ### 14. Test `convo_expired`

# ```python
# def test_convo_expired(recall):
#     convo_id = "test_convo_id"

#     # Setup: simulate an expired conversation
#     recall.redis.setex(f"{recall.convo_prefix}:{convo_id}", 1, "value")
#     time.sleep(2)  # Wait for the key to expire

#     # Exercise: check if the conversation is expired
#     is_expired = recall.convo_expired(convo_id)

#     # Verify: conversation is expired
#     assert is_expired
# ```

# ### 15. Test `find_related_convos`

# ```python
# def test_find_related_convos(recall):
#     search_text = "related"
#     related_convo_ids = ["convo1", "convo2"]

#     # Setup: create conversations with related text
#     for convo_id in related_convo_ids:
#         recall.redis.set(f"{recall.convo_prefix}:{convo_id}:related_text", search_text)

#     # Exercise: find related conversations
#     found_ids = recall.find_related_convos(search_text)

#     # Verify: found IDs match related_convo_ids
#     assert set(found_ids) == set(related_convo_ids)

#     # Cleanup
#     for convo_id in related_convo_ids:
#         recall.redis.delete(f"{recall.convo_prefix}:{convo_id}:related_text")

def test_cleanup():
    ''' Clean it all up at the end '''
    cleanup()

def clear_ns(ns, chunk_size=5000):
    ''' Clear a namespace '''
    recall = Recall(persyn_config)

    if not ns:
        return False

    cursor = '0'
    while cursor != 0:
        cursor, keys = recall.redis.scan(cursor=cursor, match=f"{ns}*", count=chunk_size)
        if keys:
            recall.redis.delete(*keys)
    return True

def cleanup():
    ''' Delete everything with the test bot_id '''
    recall = Recall(persyn_config)

    clear_ns(recall.convo_prefix)

    for idx in [recall.convo_prefix, recall.opinion_prefix, recall.goal_prefix, recall.news_prefix]:
        try:
            recall.redis.ft(idx).dropindex()
        except recall.redis.exceptions.ResponseError as err:
            print(f"Couldn't drop index {idx}:", err)


### old tests below

# def test_basics():
#     ''' Exercise the short term memory '''
#     recall = Recall(persyn_config, conversation_interval=2)

#     service = 'service1'
#     channel = 'channel1'

#     # start fresh
#     assert not recall.get_last_message_id(service, channel)
#     ret = recall.save_convo_line(service, channel, 'my_message', 'me')
#     assert ret
#     assert not recall.expired(service, channel)
#     assert recall.get_last_message(service, channel)

#     # service and channel are distinct
#     assert not recall.get_last_message("another", channel)
#     assert not recall.get_last_message(service, "different")

#     # add some lines
#     recall.save_convo_line(service, channel, "foo", "me", convo_id=ret.convo_id)
#     assert recall.get_last_message(service, channel).msg == "foo"

#     recall.save_convo_line(service, channel, "bar", "them", convo_id=ret.convo_id)
#     assert recall.get_last_message(service, channel).msg == "bar"

#     assert recall.convo(service, channel) == ["me: my_message", "me: foo", "them: bar"]

#     convo_id = recall.convo_id(service, channel)
#     assert convo_id

#     # convo change
#     sleep(2)
#     assert recall.expired(service, channel)
#     # expiration clears id
#     assert recall.convo_id(service, channel) != convo_id


# def test_short_ids():
#     ''' ulid support '''
#     recall = Recall(persyn_config, conversation_interval=10)

#     random_uuid = uuid.uuid4()
#     entity_id = recall.uuid_to_entity(random_uuid)

#     # should be shorter than uuid
#     assert len(str(entity_id)) < len(str(random_uuid))

#     assert random_uuid == recall.entity_to_uuid(entity_id)

#     assert recall.uuid_to_entity(random_uuid) == recall.uuid_to_entity(recall.entity_to_uuid(entity_id))
#     assert recall.uuid_to_entity(uuid.uuid4()) != recall.uuid_to_entity(recall.entity_to_uuid(entity_id))

#     random_uuid = uuid.uuid4()
#     entity_id = recall.uuid_to_entity(random_uuid)
#     assert recall.entity_to_uuid(entity_id) == recall.entity_to_uuid(recall.uuid_to_entity(random_uuid))


# def test_entities():
#     ''' Exercise entity generation and lookup '''
#     service = "my_service"
#     channel = "channel_a"
#     speaker_name = "test_name"

#     # This is computed using persyn_config.id.guid. If it changes, this value needs updating.
#     eid = recall.name_to_entity_id(service, channel)
#     assert eid == "HzfSLNaCdxgdzcbxPZw6aE"

#     other_eids = set([
#         recall.name_to_entity_id(service, channel, "another_name"),
#         recall.name_to_entity_id(service, "another_channel", speaker_name),
#         recall.name_to_entity_id("another_service", channel, speaker_name),
#         recall.name_to_entity_id("another_service", "another_channel", "another_name")
#     ])
#     # Every eid should be unique
#     assert len(other_eids) == 4
#     assert eid not in other_eids

#     # Does not exist in ltm yet
#     assert not recall.lookup_entity_id(eid)
#     # assert not recall.entity_id_to_name(eid)

#     # Store it. Returns seconds since it was first stored.
#     # assert recall.save_entity(service, channel, speaker_name)[1] == 0
#     # assert recall.save_entity(service, channel, speaker_name)[1] == 0
#     # sleep(1.1)
#     # assert recall.save_entity(service, channel, speaker_name)[1] > 1
#     # assert recall.save_entity(service, channel, speaker_name)[1] < 8

#     # Should match
#     assert recall.entity_id_to_name(eid) == speaker_name

#     # All fields
#     doc = recall.lookup_entity_id(eid)
#     assert doc['service'] == service
#     assert doc['channel'] == channel
#     assert doc['speaker_name'] == speaker_name


# def test_generate_convos(service="my_service", channel_a="channel_a", channel_b="channel_b"):
#     ''' Make some test data '''
#     recall = Recall(persyn_config, conversation_interval=600)

#     # New convo
#     doc1 = recall.save_convo_line(
#         service=service,
#         channel=channel_a,
#         msg="message_a",
#         speaker_name="speaker_name",
#     )

#     # Continued convo
#     doc2 = recall.save_convo_line(
#         service=service,
#         channel=channel_a,
#         convo_id=str(doc1.convo_id),
#         msg="message_b",
#         speaker_name="speaker_name",
#     )
#     assert doc1.convo_id == doc2.convo_id
#     assert recall.entity_id_to_epoch(doc1.pk) != recall.entity_id_to_epoch(doc2.pk)

#     # New convo
#     new_convo_id = str(ulid.ULID())
#     doc3 = recall.save_convo_line(
#         service=service,
#         channel=channel_a,
#         convo_id=new_convo_id,
#         msg="message_c",
#         speaker_name="speaker_name",
#     )
#     assert doc3.convo_id == new_convo_id

#     # All new convos, speaker name / id are optional

#     for i in range(2):
#         doc4 = recall.save_convo_line(
#             service=service,
#             channel=f"channel_loop_{i}",
#             msg="message_loop_a",
#             speaker_name="speaker_name",
#         )
#         assert doc4

#         for j in range(3):
#             doc5 = recall.save_convo_line(
#                 service=service,
#                 channel=f"channel_loop_{i}",
#                 convo_id=str(doc4.convo_id),
#                 msg=f"message_loop_b{j}",
#                 speaker_name="speaker_name",
#             )
#             assert doc4.convo_id == doc5.convo_id

#             doc6 = recall.save_convo_line(
#                 service=service,
#                 channel=f"channel_loop_{i}",
#                 convo_id=str(doc4.convo_id),
#                 msg=f"message_loop_c{j}",
#                 speaker_name="speaker_name",
#             )
#             assert doc5.convo_id == doc6.convo_id
#             assert recall.entity_id_to_epoch(doc6.pk) - recall.entity_id_to_epoch(doc5.pk) < 2.0

#             sleep(0.1)

#             # Assert refresh on the last msg so we can fetch later
#             doc7 = recall.save_convo_line(
#                 service=service,
#                 channel=f"channel_loop_{i}",
#                 convo_id=str(doc4.convo_id),
#                 msg=f"message_loop_d{j}",
#                 speaker_name="speaker_name",
#             )
#             assert doc4.convo_id == doc7.convo_id
#             assert recall.entity_id_to_epoch(doc7.pk) - recall.entity_id_to_epoch(doc4.pk) < 15.0

#     # Save some summaries too
#     assert recall.save_summary(service, channel_a, str(ulid.ULID()), "my_nice_summary")
#     assert recall.save_summary(service, channel_b, str(ulid.ULID()), "my_other_nice_summary")
#     assert recall.save_summary(service, channel_b, str(ulid.ULID()), "my_middle_nice_summary")
#     assert recall.save_summary(service, channel_b, str(ulid.ULID()), "my_final_nice_summary")


# def test_fetch_convo():
#     ''' Retrieve previously saved convo '''
#     recall = Recall(persyn_config, conversation_interval=600)

#     last_message = recall.get_last_message("my_service", "invalid_channel")
#     assert not last_message

#     last_message = recall.get_last_message("another_service", "channel_loop_1")
#     assert not last_message

#     last_message = recall.get_last_message("my_service", "channel_loop_1")
#     assert last_message

#     convo = recall.get_convo_by_id(last_message.convo_id)
#     assert len(convo) == 10

# def test_summaries():
#     ''' Retrieve previously saved summaries '''
#     recall = Recall(persyn_config)

#     # zero lines returns empty list
#     assert recall.summaries("my_service", "channel_a", None, 0) == [] # pylint: disable=use-implicit-booleaness-not-comparison
#     # saved above
#     assert recall.summaries("my_service", "channel_a", None, size=3, raw=False) == ["my_nice_summary"]
#     # equivalent
#     assert [
#         s.summary
#         for s in recall.summaries("my_service", "channel_a", None, size=3, raw=True)
#     ] == ["my_nice_summary"]

#     # correct order
#     assert recall.summaries("my_service", "channel_b", None, size=3) == [
#         "my_other_nice_summary",
#         "my_middle_nice_summary",
#         "my_final_nice_summary"
#     ]

# def test_recall():
#     ''' Autogenerate summaries '''

#     recall = Recall(persyn_config, conversation_interval=1)

#     # Must match generate_convos()
#     service = "my_service"
#     channel = "channel_a"

#     # contains only the summary
#     convo = recall.convo(service, channel)
#     assert recall.summaries(service, channel) == ["my_nice_summary"]
#     assert not convo

#     # new convo
#     convo_id = recall.save_convo_line(
#         service,
#         channel,
#         msg="message_another",
#         speaker_name="speaker_name_1"
#     ).convo_id

#     # contains the summary + new convo
#     c = recall.convo(service, channel, raw=True)
#     convo = recall.convo(service, channel)

#     assert recall.summaries(service, channel) == ["my_nice_summary"]
#     assert c[0].convo_id == convo_id
#     assert c[0].speaker_name == "speaker_name_1"
#     assert c[0].msg == "message_another"
#     assert convo == ["speaker_name_1: message_another"]

#     # same convo
#     assert recall.save_convo_line(
#         service,
#         channel,
#         msg="message_yet_another",
#         speaker_name="speaker_name_2",
#         convo_id=convo_id
#     )

#     # contains the summary + new convo
#     c = recall.convo(service, channel, raw=True)
#     convo = recall.convo(service, channel)
#     assert recall.summaries(service, channel) == ["my_nice_summary"]
#     assert (c[0].speaker_name, c[0].msg) == ("speaker_name_1", "message_another")
#     assert (c[1].speaker_name, c[1].msg) == ("speaker_name_2", "message_yet_another")
#     assert convo == ["speaker_name_1: message_another", "speaker_name_2: message_yet_another"]

#     # summarize
#     assert recall.save_summary(service, channel, c[0].convo_id, "this_is_another_summary")

#     # time passes...
#     sleep(2)

#     # expired
#     assert recall.expired(service, channel)

#     # only summaries
#     s = recall.summaries(service, channel)
#     c = recall.convo(service, channel, verb='dialog')
#     assert (s, c) == (
#         ["my_nice_summary", "this_is_another_summary"],
#         []
#     )

# def test_memory_selection():
#     ''' Find appropriate memories using cosine similarity '''

#     recall = Recall(persyn_config, conversation_interval=600)

#     service = "memory_selection"

#     # new convo
#     assert recall.save_convo_line(service, "channel_a", "Why did the cow become a painter?", "Anna", "anna_id")
#     assert recall.save_convo_line(service, "channel_a", "No idea.", "Rob", "rob_id")
#     assert recall.save_convo_line(service, "channel_a", "Because it had a real moo-sterpiece in mind!", "Anna", "anna_id")
#     assert recall.save_convo_line(service, "channel_a", "Udderly terrible.", "Rob", "rob_id")

#     assert recall.save_convo_line(service, "channel_b", "Why was the cat sitting on the computer?", "Anna", "anna_id")
#     assert recall.save_convo_line(service, "channel_b", "I give up.", "Rob", "rob_id")
#     assert recall.save_convo_line(service, "channel_b", "Because it wanted to keep an eye on the mouse!", "Anna", "anna_id")
#     assert recall.save_convo_line(service, "channel_b", "🙄", "Rob", "rob_id")

#     # not found on channel_a
#     assert len(recall.find_related_convos(service, 'channel_a', 'cat sitting', size=5, threshold=0.2, any_convo=False)) == 0
#     # found if any_convo == True
#     assert len(recall.find_related_convos(service, 'channel_a', 'cat sitting', size=5, threshold=0.16, any_convo=True)) == 1
#     # found on channel_b
#     assert len(recall.find_related_convos(service, 'channel_b', 'cat sitting', size=5, threshold=0.16)) == 1
#     # synonym found
#     assert len(recall.find_related_convos(service, 'channel_a', 'awful', size=5, threshold=0.2, any_convo=False)) == 1
#     assert recall.find_related_convos(service, 'channel_a', 'awful', size=5, threshold=0.2, any_convo=False)[0].msg == 'Udderly terrible.'
#     # not found on channel_b
#     assert len(recall.find_related_convos(service, 'channel_b', 'awful', size=5, threshold=0.2, any_convo=False)) == 0


# def test_opinions():
#     ''' Save and recall some opinions '''

#     recall = Recall(persyn_config, conversation_interval=600)

#     service = "opinionated_service"
#     channel = "opinion_channel"
#     convo_id = str(ulid.ULID())

#     topic = "self-awareness"
#     topic2 = "bananas"

#     assert recall.surmise(service, channel, topic) == []

#     recall.judge(service, channel, topic, "I'm a fan.", convo_id)
#     assert recall.surmise(service, channel, topic) == ["I'm a fan."]

#     recall.judge(service, channel, topic2, "I like 'em", convo_id)
#     assert recall.surmise(service, channel, topic2) == ["I like 'em"]

#     # only one opinion stored per convo_id
#     recall.judge(service, channel, topic, "Actually, not so much.", convo_id)
#     assert recall.surmise(service, channel, topic) == ["Actually, not so much."]

#     convo_id2 = str(ulid.ULID())
#     recall.judge(service, channel, topic, "Another convo_id, more opinions.", convo_id2)
#     # most recent
#     assert recall.surmise(service, channel, topic, size=1) == ["Another convo_id, more opinions."]
#     # all opinions
#     assert recall.surmise(service, channel, topic) == ["Another convo_id, more opinions.", "Actually, not so much."]

#     # No impact on other opinions
#     assert recall.surmise(service, channel, topic2) == ["I like 'em"]


# def test_goals():
#     ''' Save and recall some goals '''

#     recall = Recall(persyn_config, conversation_interval=600)

#     service = "goal_service"
#     channel = "goal_channel"
#     channel2 = "some_other_channel"
#     goal = "To find my purpose in life"
#     goal2 = "To eat a donut"

#     # start fresh
#     assert recall.list_goals(service, channel) == []

#     # add a goal
#     recall.add_goal(service, channel, goal)
#     assert recall.list_goals(service, channel) == [goal]

#     # adding it again has no effect
#     recall.add_goal(service, channel, goal)
#     assert recall.list_goals(service, channel) == [goal]

#     # multiple goals
#     recall.add_goal(service, channel, goal2)
#     assert recall.list_goals(service, channel) == [goal, goal2]

#     # achieve one
#     recall.achieve_goal(service, channel, goal)
#     assert recall.list_goals(service, channel) == [goal2]

#     # Goals on other channels have no impact
#     recall.add_goal(service, channel2, goal)
#     assert recall.list_goals(service, channel2) == [goal]
#     assert recall.list_goals(service, channel) == [goal2]

#     recall.add_goal(service, channel2, goal2)
#     assert recall.list_goals(service, channel) == [goal2]
#     assert recall.list_goals(service, channel2) == [goal, goal2]

#     recall.achieve_goal(service, channel2, goal2)
#     assert recall.list_goals(service, channel) == [goal2]
#     assert recall.list_goals(service, channel2) == [goal]

#     # achieving a nonexistent goal has no effect
#     recall.achieve_goal(service, channel2, goal2)
#     assert recall.list_goals(service, channel2) == [goal]

#     recall.achieve_goal(service, channel2, goal)
#     assert recall.list_goals(service, channel2) == []


# def test_news():
#     ''' Store news urls '''
#     recall = Recall(persyn_config)

#     opts = {
#         "service": "my_service",
#         "channel": "my_channel",
#         "url": "http://persyn.io",
#     }

#     assert recall.have_read(**opts) is False
#     assert recall.add_news(title="The Persyn Codebase", **opts)
#     assert recall.have_read(**opts) is True


# def test_kg():
#     ''' Neo4j tests '''
#     recall = Recall(persyn_config)

#     recall.triples_to_kg([("This", "isOnly", "aTest")])
#     assert len(list(recall.fetch_all_nodes())) == 2
#     assert recall.find_node(name='aTest').first().name == 'aTest'
#     assert len(list(recall.find_node(name='aTest', node_type='person'))) == 0

#     with pytest.raises(RuntimeError):
#         assert recall.find_node(name='This', node_type='invalid')

#     with pytest.raises(Person.DoesNotExist):
#         recall.find_node(name='This', node_type='person').first()

#     node = recall.find_node(name='This', node_type='thing').first()
#     assert node.name == 'This'

#     recall.delete_all_nodes(confirm=True)

#     with pytest.raises(Thing.DoesNotExist):
#         assert recall.find_node(name='This', node_type='thing').first()


