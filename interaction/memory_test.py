'''
memory (elasticsearch) tests
'''
import os
import datetime as dt
import uuid
import sys

from pathlib import Path
from time import sleep

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../').resolve()))

from memory import LongTermMemory, ShortTermMemory, Recall

# Bot config
from utils.config import load_config

persyn_config = load_config()

prefix = f"{persyn_config.id.name.lower()}-test"

# Dynamic test index names
now = dt.datetime.now().isoformat().replace(':','.').lower()

ltm = LongTermMemory(
    bot_name=persyn_config.id.name,
    bot_id=persyn_config.id.guid,
    url=persyn_config.memory.elastic.url,
    auth_name=persyn_config.memory.elastic.user,
    auth_key=persyn_config.memory.elastic.key,
    index_prefix=prefix,
    version=now,
    verify_certs=True
)

recall = Recall(
    bot_name=persyn_config.id.name,
    bot_id=persyn_config.id.guid,
    url=persyn_config.memory.elastic.url,
    auth_name=persyn_config.memory.elastic.user,
    auth_key=persyn_config.memory.elastic.key,
    index_prefix=prefix,
    version=now,
    conversation_interval=0.5,
    verify_certs=True
)

def test_stm():
    ''' Exercise the short term memory '''
    stm = ShortTermMemory(conversation_interval=0.1)

    service = 'service1'
    channel = 'channel1'

    # start fresh
    assert not stm.exists(service, channel)
    assert stm.expired(service, channel)
    stm.create(service, channel)
    assert stm.exists(service, channel)
    assert not stm.expired(service, channel)

    # service and channel are distinct
    assert not stm.exists("another", channel)
    assert not stm.exists(service, "different")

    # add some lines
    stm.append(service, channel, "foo")
    assert stm.last(service, channel) == "foo"

    stm.append(service, channel, "bar")
    assert stm.last(service, channel) == "bar"

    assert stm.fetch(service, channel) == ["foo", "bar"]

    convo_id = stm.convo_id(service, channel)
    assert convo_id

    # convo change
    sleep(0.2)
    assert stm.expired(service, channel)
    # expiration does not clear id
    assert stm.convo_id(service, channel) == convo_id

    # append does clear id
    stm.append(service, channel, "bar")
    assert stm.last(service, channel) == "bar"
    new_id = stm.convo_id(service, channel)
    assert new_id != convo_id

    # explicit clear
    stm.clear(service, channel)
    assert not stm.expired(service, channel)
    assert stm.convo_id(service, channel) != new_id
    assert stm.last(service, channel) is None

def test_short_ids():
    ''' shortuuid support '''
    random_uuid = uuid.uuid4()
    entity_id = ltm.uuid_to_entity(random_uuid)
    assert str(random_uuid) == ltm.entity_to_uuid(entity_id)

    entity_id = ltm.uuid_to_entity(str(random_uuid))
    assert str(random_uuid) == ltm.entity_to_uuid(entity_id)

def test_entities():
    ''' Exercise entity generation and lookup '''
    service = "my_service"
    channel = "channel_a"
    speaker_name = "test_name"
    speaker_id = "test_id"

    eid = ltm.name_to_entity(service, channel, speaker_id)
    assert eid == "j6GhcuBe5FAPRtNsdASut5"

    other_eids = set([
        ltm.name_to_entity(service, channel, "another_name"),
        ltm.name_to_entity(service, "another_channel", speaker_name),
        ltm.name_to_entity("another_service", channel, speaker_name),
        ltm.name_to_entity("another_service", "another_channel", "another_name")
    ])
    # Every eid should be unique
    assert len(other_eids) == 4
    assert eid not in other_eids

    # Does not exist in ltm yet
    assert not ltm.lookup_entity(eid)
    assert not ltm.entity_to_name(eid)

    # Store it. Returns seconds since it was first stored.
    assert ltm.save_entity(service, channel, speaker_name, speaker_id)[1] == 0
    assert ltm.save_entity(service, channel, speaker_name)[1] == 0
    sleep(1.1)
    assert ltm.save_entity(service, channel, speaker_name, speaker_id)[1] > 1
    assert ltm.save_entity(service, channel, speaker_name, speaker_id)[1] < 2

    # Should match
    assert ltm.entity_to_name(eid) == speaker_name

    # All fields
    doc = ltm.lookup_entity(eid)
    assert doc['service'] == service
    assert doc['channel'] == channel
    assert doc['speaker_name'] == speaker_name
    assert doc['speaker_id'] == speaker_id

def test_save_convo():
    ''' Make some test data '''

    # New convo
    cid1, ts1 = ltm.save_convo("my_service", "channel_a", "message_a", "speaker_name", "speaker_id")
    assert cid1
    assert ts1
    # Continued convo
    cid2, ts2 = ltm.save_convo("my_service", "channel_a", "message_b", "speaker_name", "speaker_id", convo_id=cid1)
    assert cid1 == cid2
    assert ts2
    # New convo
    cid3, ts3 = ltm.save_convo("my_service", "channel_a", "message_b", "speaker_name", "speaker_id", convo_id="foo")
    assert cid3 == "foo"
    assert ts3

    # All new convos, speaker name / id are optional

    for i in range(2):
        cid0, ts0 = ltm.save_convo(
            "my_service",
            f"channel_loop_{i}",
            "message_loop_a",
            "speaker_name",
            "speaker_id",
            convo_id=None
        )
        assert ts0

        for j in range(3):
            cid1, ts1 = ltm.save_convo(
                "my_service",
                f"channel_loop_{i}",
                f"message_loop_b{j}",
                speaker_id="speaker_id",
                convo_id=cid0)
            assert cid1 == cid0

            cid2, ts2 = ltm.save_convo(
                "my_service",
                f"channel_loop_{i}",
                f"message_loop_c{j}",
                speaker_name="speaker_name",
                convo_id=cid0)
            assert cid1 == cid2
            assert ts1 != ts2

            # Assert refresh on the last msg so we can fetch later
            cid3, ts3 = ltm.save_convo(
                "my_service",
                f"channel_loop_{i}",
                f"message_loop_d{j}",
                convo_id=cid1,
                refresh=True
            )
            assert cid1 == cid3
            assert ts1 != ts3

def test_fetch_convo():
    ''' Retrieve previously saved convo '''
    assert len(ltm.load_convo("my_service", "channel_loop_0")) == 10
    assert len(ltm.load_convo("my_service", "channel_loop_0", lines=3)) == 3
    # First message (whole convo)
    assert ltm.load_convo("my_service", "channel_loop_0")[0] == "speaker_name: message_loop_a"
    # Last message (most recent 1 line)
    assert ltm.load_convo("my_service", "channel_loop_0", lines=1)[0] == "None: message_loop_d2"

    last_message = ltm.get_last_message("my_service", "invalid_channel")
    assert not last_message

    last_message = ltm.get_last_message("another_service", "channel_loop_1")
    assert not last_message

    last_message = ltm.get_last_message("my_service", "channel_loop_1")
    assert last_message

    convo = ltm.get_convo_by_id(last_message['_source']['convo_id'])
    assert len(convo) == 10

def test_save_summaries():
    ''' Make some test data '''
    service = "my_service"
    channel_a = "channel_a"
    channel_b = "channel_b"

    assert ltm.save_summary(service, channel_a, "convo_id", "my_nice_summary")
    assert ltm.save_summary(service, channel_b, "convo_id_2", "my_other_nice_summary")
    assert ltm.save_summary(service, channel_b, "convo_id_3", "my_middle_nice_summary")
    assert ltm.save_summary(service, channel_b, "convo_id_4", "my_final_nice_summary", refresh=True)

def test_load_summaries():
    ''' Retrieve previously saved summaries '''

    # zero lines returns empty list
    assert ltm.load_summaries("my_service", "channel_a", 0) == [] # pylint: disable=use-implicit-booleaness-not-comparison
    # saved above
    assert ltm.load_summaries("my_service", "channel_a") == ["my_nice_summary"]
    # correct order
    assert ltm.load_summaries("my_service", "channel_b") == [
        "my_other_nice_summary",
        "my_middle_nice_summary",
        "my_final_nice_summary"
    ]

def test_recall():
    ''' Use stm + ltm together to autogenerate summaries '''
    # Must match test_save_summaries()
    service = "my_service"
    channel = "channel_a"

    # contains only the summary
    assert recall.load(service, channel) == (['my_nice_summary'], [])

    # new convo
    assert recall.save(service, channel, "message_another", "speaker_name_2", "speaker_id")

    # contains the summary + new convo
    assert recall.load(service, channel) == (
        ["my_nice_summary"],
        ["speaker_name_2: message_another"]
    )

    # same convo
    assert recall.save(service, channel, "message_yet_another", "speaker_name_1", "speaker_id")

    # contains the summary + new convo
    assert recall.load(service, channel) == (
        ["my_nice_summary"],
        ["speaker_name_2: message_another", "speaker_name_1: message_yet_another"]
    )

    # summarize
    assert recall.summary(service, channel, "this_is_another_summary")

    # time passes...
    sleep(0.6)

    # expired
    assert recall.expired(service, channel)

    # only summaries
    assert recall.load(service, channel) == (
        ["my_nice_summary", "this_is_another_summary"],
        []
    )

def test_cleanup():
    ''' Delete indices '''
    for i in ltm.index.items():
        ltm.es.options(ignore_status=[400, 404]).indices.delete(index=i[1]) # pylint: disable=unexpected-keyword-arg
