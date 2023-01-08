'''
memory (elasticsearch) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, no-member
import datetime as dt
import uuid

from time import sleep
from copy import copy

from interaction.chrono import elapsed

from interaction.memory import LongTermMemory, ShortTermMemory, Recall

# Bot config
from utils.config import load_config

# from utils.color_logging import log

persyn_config = load_config()

prefix = f"{persyn_config.id.name.lower()}-test"

# Dynamic test index names
now = dt.datetime.now().isoformat().replace(':','.').lower()

ltm = LongTermMemory(persyn_config, version=now)

def test_stm():
    ''' Exercise the short term memory '''
    stm = ShortTermMemory(persyn_config, conversation_interval=0.01)

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

    # This is computed using persyn_config.id.guid. If it changes, this value needs updating.
    eid = ltm.name_to_entity(service, channel, speaker_id)
    assert eid == "HzfSLNaCdxgdzcbxPZw6aE"

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
    doc1 = ltm.save_convo("my_service", "channel_a", "message_a", "speaker_name", "speaker_id")
    assert 'convo_id' in doc1
    assert '@timestamp' in doc1
    # Continued convo
    doc2 = ltm.save_convo(
        "my_service", "channel_a", "message_b",
        "speaker_name", "speaker_id", convo_id=doc1['convo_id']
    )
    assert doc1['convo_id'] == doc2['convo_id']
    assert doc1['@timestamp'] != doc2['@timestamp']
    # New convo
    doc3 = ltm.save_convo("my_service", "channel_a", "message_b", "speaker_name", "speaker_id", convo_id="foo")
    assert doc3['convo_id'] == "foo"

    # All new convos, speaker name / id are optional

    for i in range(2):
        doc4 = ltm.save_convo(
            "my_service",
            f"channel_loop_{i}",
            "message_loop_a",
            "speaker_name",
            "speaker_id",
            convo_id=None
        )
        assert doc4

        for j in range(3):
            doc5 = ltm.save_convo(
                "my_service",
                f"channel_loop_{i}",
                f"message_loop_b{j}",
                speaker_id="speaker_id",
                convo_id=doc4['convo_id'])
            assert doc4['convo_id'] == doc5['convo_id']

            doc6 = ltm.save_convo(
                "my_service",
                f"channel_loop_{i}",
                f"message_loop_c{j}",
                speaker_name="speaker_name",
                convo_id=doc4['convo_id'])
            assert doc5['convo_id'] == doc6['convo_id']
            assert elapsed(doc5['@timestamp'], doc6['@timestamp']) < 1.0

            sleep(0.1)

            # Assert refresh on the last msg so we can fetch later
            doc7 = ltm.save_convo(
                "my_service",
                f"channel_loop_{i}",
                f"message_loop_d{j}",
                convo_id=doc4['convo_id'],
                refresh=True
            )
            assert doc4['convo_id'] == doc7['convo_id']
            assert elapsed(doc4['@timestamp'], doc7['@timestamp']) > 0.1
            assert elapsed(doc4['@timestamp'], doc7['@timestamp']) < 5.0

def test_fetch_convo():
    ''' Retrieve previously saved convo '''
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

def test_lookup_summaries():
    ''' Retrieve previously saved summaries '''

    # zero lines returns empty list
    assert ltm.lookup_summaries("my_service", "channel_a", None, 0) == [] # pylint: disable=use-implicit-booleaness-not-comparison
    # saved above
    assert [
        s['_source']['summary']
        for s in ltm.lookup_summaries("my_service", "channel_a", None, size=3)
    ] == ["my_nice_summary"]

    # correct order
    assert [s['_source']['summary'] for s in ltm.lookup_summaries("my_service", "channel_b", None, size=3)] == [
        "my_other_nice_summary",
        "my_middle_nice_summary",
        "my_final_nice_summary"
    ]

def test_recall():
    ''' Use stm + ltm together to autogenerate summaries '''

    recall = Recall(persyn_config, version=now, conversation_interval=0.5)

    # Must match test_save_summaries()
    service = "my_service"
    channel = "channel_a"

    # contains only the summary
    s = recall.summaries(service, channel)
    c = recall.stm.fetch(service, channel)
    convo = recall.convo(service, channel)
    assert (s, c) == (["my_nice_summary"], [])
    assert not convo

    # new convo
    assert recall.save(service, channel, "message_another", "speaker_name_1", "speaker_id")

    # contains the summary + new convo
    s = recall.summaries(service, channel)
    c = recall.stm.fetch(service, channel)
    convo = recall.convo(service, channel)

    assert s == ["my_nice_summary"]
    assert c[0]['speaker'] == "speaker_name_1"
    assert c[0]['msg'] == "message_another"
    assert convo == ["speaker_name_1: message_another"]

    # same convo
    assert recall.save(service, channel, "message_yet_another", "speaker_name_2", "speaker_id")

    # contains the summary + new convo
    s = recall.summaries(service, channel)
    c = recall.stm.fetch(service, channel)
    convo = recall.convo(service, channel)
    assert s == ["my_nice_summary"]
    assert (c[0]['speaker'], c[0]['msg']) == ("speaker_name_1", "message_another")
    assert (c[1]['speaker'], c[1]['msg']) == ("speaker_name_2", "message_yet_another")
    assert convo == ["speaker_name_1: message_another", "speaker_name_2: message_yet_another"]

    # summarize
    assert recall.summary(service, channel, "this_is_another_summary")

    # time passes...
    sleep(0.6)

    # expired
    assert recall.expired(service, channel)

    # only summaries
    s = recall.summaries(service, channel)
    c = recall.stm.fetch(service, channel)
    assert (s, c) == (
        ["my_nice_summary", "this_is_another_summary"],
        []
    )

def test_relationships():
    ''' Store and retrieve relationships '''

    opts = {
        "service": "my_service",
        "channel": "my_channel",
        "speaker_id": "a_speaker_id",
        "source_id": "some_random_source",
        "rel": "testing",
        "target_id": "another_target_id",
        "convo_id": "boring_conversation",
        "graph": {"nodes": [1, 2, 3], "edges": [{"source": 1, "target": 2, "edge": "connected"}]}
    }

    assert ltm.save_relationship(**opts)['result'] == 'created'

    q = copy(opts)
    del q['graph']

    # exact match
    ret = ltm.lookup_relationship(**q)[0]['_source']
    del ret['@timestamp']
    assert ret == opts

    # negative match
    assert ltm.lookup_relationship(
        service=opts['service'],
        channel=opts['channel'],
        foo='bar'
    ) == []

    # test partial matches
    for k in ['source_id', 'rel', 'target_id']:
        del q[k]

    ret = ltm.lookup_relationship(**q)[0]['_source']
    del ret['@timestamp']
    assert ret == opts

    del q['convo_id']

    ret = ltm.lookup_relationship(**q)[0]['_source']
    del ret['@timestamp']
    assert ret == opts

def test_cleanup():
    ''' Delete indices '''
    for i in ltm.index.items():
        ltm.es.options(ignore_status=[400, 404]).indices.delete(index=i[1])
