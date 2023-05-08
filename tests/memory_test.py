'''
memory (redis) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, no-member
import uuid

from time import sleep

import ulid

from src.persyn.interaction.memory import LongTermMemory, ShortTermMemory, Recall

# Bot config
from src.persyn.utils.config import load_config

# from utils.color_logging import log

persyn_config = load_config()

ltm = LongTermMemory(persyn_config)

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
    ''' ulid support '''
    random_uuid = uuid.uuid4()
    entity_id = ltm.uuid_to_entity(random_uuid)

    # should be shorter than uuid
    assert len(str(entity_id)) < len(str(random_uuid))

    assert random_uuid == ltm.entity_to_uuid(entity_id)

    assert ltm.uuid_to_entity(random_uuid) == ltm.uuid_to_entity(ltm.entity_to_uuid(entity_id))
    assert ltm.uuid_to_entity(uuid.uuid4()) != ltm.uuid_to_entity(ltm.entity_to_uuid(entity_id))

    random_uuid = uuid.uuid4()
    entity_id = ltm.uuid_to_entity(random_uuid)
    assert ltm.entity_to_uuid(entity_id) == ltm.entity_to_uuid(ltm.uuid_to_entity(random_uuid))


# def test_entities():
#     ''' Exercise entity generation and lookup '''
#     service = "my_service"
#     channel = "channel_a"
#     speaker_name = "test_name"
#     speaker_id = "test_id"

#     # This is computed using persyn_config.id.guid. If it changes, this value needs updating.
#     eid = ltm.name_to_entity_id(service, channel, speaker_id)
#     assert eid == "HzfSLNaCdxgdzcbxPZw6aE"

#     other_eids = set([
#         ltm.name_to_entity_id(service, channel, "another_name"),
#         ltm.name_to_entity_id(service, "another_channel", speaker_name),
#         ltm.name_to_entity_id("another_service", channel, speaker_name),
#         ltm.name_to_entity_id("another_service", "another_channel", "another_name")
#     ])
#     # Every eid should be unique
#     assert len(other_eids) == 4
#     assert eid not in other_eids

#     # Does not exist in ltm yet
#     assert not ltm.lookup_entity_id(eid)
#     # assert not ltm.entity_id_to_name(eid)

#     # Store it. Returns seconds since it was first stored.
#     # assert ltm.save_entity(service, channel, speaker_name, speaker_id)[1] == 0
#     # assert ltm.save_entity(service, channel, speaker_name)[1] == 0
#     # sleep(1.1)
#     # assert ltm.save_entity(service, channel, speaker_name, speaker_id)[1] > 1
#     # assert ltm.save_entity(service, channel, speaker_name, speaker_id)[1] < 8

#     # Should match
#     assert ltm.entity_id_to_name(eid) == speaker_name

#     # All fields
#     doc = ltm.lookup_entity_id(eid)
#     assert doc['service'] == service
#     assert doc['channel'] == channel
#     assert doc['speaker_name'] == speaker_name
#     assert doc['speaker_id'] == speaker_id

def test_save_convo():
    ''' Make some test data '''

    # New convo
    doc1 = ltm.save_convo(
        service="my_service",
        channel="channel_a",
        msg="message_a",
        speaker_id="speaker_id",
        speaker_name="speaker_name",
    )

    # Continued convo
    doc2 = ltm.save_convo(
        service="my_service",
        channel="channel_a",
        convo_id=str(doc1.convo_id),
        msg="message_b",
        speaker_id="speaker_id",
        speaker_name="speaker_name",
    )
    assert doc1.convo_id == doc2.convo_id
    assert ltm.entity_id_to_timestamp(doc1.pk) != ltm.entity_id_to_timestamp(doc2.pk)

    # New convo
    doc3 = ltm.save_convo(
        service="my_service",
        channel="channel_a",
        convo_id="foo",
        msg="message_b",
        speaker_id="speaker_id",
        speaker_name="speaker_name",
    )
    assert doc3.convo_id == "foo"

    # All new convos, speaker name / id are optional

    for i in range(2):
        doc4 = ltm.save_convo(
            service="my_service",
            channel=f"channel_loop_{i}",
            msg="message_loop_a",
            speaker_id="speaker_id",
            speaker_name="speaker_name",
        )
        assert doc4

        for j in range(3):
            doc5 = ltm.save_convo(
                service="my_service",
                channel=f"channel_loop_{i}",
                convo_id=str(doc4.convo_id),
                msg=f"message_loop_b{j}",
                speaker_id="speaker_id",
                speaker_name="speaker_name",
            )
            assert doc4.convo_id == doc5.convo_id

            doc6 = ltm.save_convo(
                service="my_service",
                channel=f"channel_loop_{i}",
                convo_id=str(doc4.convo_id),
                msg=f"message_loop_c{j}",
                speaker_id="speaker_id",
                speaker_name="speaker_name",
            )
            assert doc5.convo_id == doc6.convo_id
            assert ltm.entity_id_to_timestamp(doc6.pk) - ltm.entity_id_to_timestamp(doc5.pk) < 2.0

            sleep(0.1)

            # Assert refresh on the last msg so we can fetch later
            doc7 = ltm.save_convo(
                service="my_service",
                channel=f"channel_loop_{i}",
                convo_id=str(doc4.convo_id),
                msg=f"message_loop_d{j}",
                speaker_id="speaker_id",
                speaker_name="speaker_name",
            )
            assert doc4.convo_id == doc7.convo_id
            assert ltm.entity_id_to_timestamp(doc7.pk) - ltm.entity_id_to_timestamp(doc4.pk) < 15.0

def test_fetch_convo():
    ''' Retrieve previously saved convo '''
    last_message = ltm.get_last_message("my_service", "invalid_channel")
    assert not last_message

    last_message = ltm.get_last_message("another_service", "channel_loop_1")
    assert not last_message

    last_message = ltm.get_last_message("my_service", "channel_loop_1")
    assert last_message

    convo = ltm.get_convo_by_id(last_message.convo_id)
    assert len(convo) == 10

def test_save_summaries():
    ''' Make some test data '''
    service = "my_service"
    channel_a = "channel_a"
    channel_b = "channel_b"

    assert ltm.save_summary(service, channel_a, "convo_id", "my_nice_summary")
    assert ltm.save_summary(service, channel_b, "convo_id_2", "my_other_nice_summary")
    assert ltm.save_summary(service, channel_b, "convo_id_3", "my_middle_nice_summary")
    assert ltm.save_summary(service, channel_b, "convo_id_4", "my_final_nice_summary")

def test_lookup_summaries():
    ''' Retrieve previously saved summaries '''

    # zero lines returns empty list
    assert ltm.lookup_summaries("my_service", "channel_a", None, 0) == [] # pylint: disable=use-implicit-booleaness-not-comparison
    # saved above
    assert [
        s.summary
        for s in ltm.lookup_summaries("my_service", "channel_a", None, size=3)
    ] == ["my_nice_summary"]

    # correct order
    assert [s.summary for s in ltm.lookup_summaries("my_service", "channel_b", None, size=3)] == [
        "my_other_nice_summary",
        "my_middle_nice_summary",
        "my_final_nice_summary"
    ]

def test_recall():
    ''' Use stm + ltm together to autogenerate summaries '''

    recall = Recall(persyn_config, conversation_interval=3)

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
    assert c[0].speaker_name == "speaker_name_1"
    assert c[0].msg == "message_another"
    assert convo == ["speaker_name_1: message_another"]

    # same convo
    assert recall.save(service, channel, "message_yet_another", "speaker_name_2", "speaker_id")

    # contains the summary + new convo
    s = recall.summaries(service, channel)
    c = recall.stm.fetch(service, channel)
    convo = recall.convo(service, channel)
    assert s == ["my_nice_summary"]
    assert (c[0].speaker_name, c[0].msg) == ("speaker_name_1", "message_another")
    assert (c[1].speaker_name, c[1].msg) == ("speaker_name_2", "message_yet_another")
    assert convo == ["speaker_name_1: message_another", "speaker_name_2: message_yet_another"]

    # summarize
    assert recall.summary(service, channel, "this_is_another_summary")

    print("SLEEPING FOR 4")
    # time passes...
    sleep(4)

    # expired
    assert recall.expired(service, channel)

    # only summaries
    s = recall.summaries(service, channel)
    c = recall.stm.fetch(service, channel)
    assert (s, c) == (
        ["my_nice_summary", "this_is_another_summary"],
        []
    )

def test_opinions():
    ''' Save and recall some opinions '''

    recall = Recall(persyn_config, conversation_interval=600)

    service = "opinionated_service"
    channel = "opinion_channel"
    convo_id = str(ulid.ULID())

    topic = "self-awareness"
    topic2 = "bananas"

    assert recall.surmise(service, channel, topic) == []

    recall.judge(service, channel, topic, "I'm a fan.", convo_id)
    assert recall.surmise(service, channel, topic) == ["I'm a fan."]

    recall.judge(service, channel, topic2, "I like 'em", convo_id)
    assert recall.surmise(service, channel, topic2) == ["I like 'em"]

    # only one opinion stored per convo_id
    recall.judge(service, channel, topic, "Actually, not so much.", convo_id)
    assert recall.surmise(service, channel, topic) == ["Actually, not so much."]

    convo_id2 = str(ulid.ULID())
    recall.judge(service, channel, topic, "Another convo_id, more opinions.", convo_id2)
    # most recent
    assert recall.surmise(service, channel, topic, size=1) == ["Another convo_id, more opinions."]
    # all opinions
    assert recall.surmise(service, channel, topic) == ["Another convo_id, more opinions.", "Actually, not so much."]

    # No impact on other opinions
    assert recall.surmise(service, channel, topic2) == ["I like 'em"]


def test_goals():
    ''' Save and recall some goals '''

    recall = Recall(persyn_config, conversation_interval=600)

    service = "goal_service"
    channel = "goal_channel"
    channel2 = "some_other_channel"
    goal = "To find my purpose in life"
    goal2 = "To eat a donut"

    # start fresh
    assert recall.ltm.list_goals(service, channel) == []

    # add a goal
    recall.ltm.add_goal(service, channel, goal)
    assert recall.ltm.list_goals(service, channel) == [goal]

    # adding it again has no effect
    recall.ltm.add_goal(service, channel, goal)
    assert recall.ltm.list_goals(service, channel) == [goal]

    # multiple goals
    recall.ltm.add_goal(service, channel, goal2)
    assert recall.ltm.list_goals(service, channel) == [goal, goal2]

    # achieve one
    recall.ltm.achieve_goal(service, channel, goal)
    assert recall.ltm.list_goals(service, channel) == [goal2]

    # Goals on other channels have no impact
    recall.ltm.add_goal(service, channel2, goal)
    assert recall.ltm.list_goals(service, channel2) == [goal]
    assert recall.ltm.list_goals(service, channel) == [goal2]

    recall.ltm.add_goal(service, channel2, goal2)
    assert recall.ltm.list_goals(service, channel) == [goal2]
    assert recall.ltm.list_goals(service, channel2) == [goal, goal2]

    recall.ltm.achieve_goal(service, channel2, goal2)
    assert recall.ltm.list_goals(service, channel) == [goal2]
    assert recall.ltm.list_goals(service, channel2) == [goal]

    # achieving a nonexistent goal has no effect
    recall.ltm.achieve_goal(service, channel2, goal2)
    assert recall.ltm.list_goals(service, channel2) == [goal]

    recall.ltm.achieve_goal(service, channel2, goal)
    assert recall.ltm.list_goals(service, channel2) == []


def test_news():
    ''' Store news urls '''

    opts = {
        "service": "my_service",
        "channel": "my_channel",
        "url": "http://persyn.io",
    }

    assert ltm.have_read(**opts) is False
    assert ltm.add_news(title="The Persyn Codebase", **opts)
    assert ltm.have_read(**opts) is True

# def test_kg():
#     ''' Neo4j tests '''
#     ltm.triples_to_kg([("This", "isOnly", "aTest")])
#     assert len(list(ltm.fetch_all_nodes())) == 2
#     assert ltm.find_node(name='aTest').first().name == 'aTest'
#     assert len(list(ltm.find_node(name='aTest', node_type='person'))) == 0

#     # with pytest.raises(Person.DoesNotExist):
#     #     ltm.find_node(name='This', node_type='person').first()

#     assert ltm.find_node(name='This', node_type='thing').first().name == 'This'

#     with pytest.raises(RuntimeError):
#         assert ltm.find_node(name='This', node_type='invalid')

def clear_ns(ns, chunk_size=5000):
    ''' Clear a namespace '''
    cursor = '0'
    while cursor != 0:
        cursor, keys = ltm.redis.scan(cursor=cursor, match=f"{ns}*", count=chunk_size)
        if keys:
            ltm.redis.delete(*keys)

def test_cleanup():
    ''' Delete everything with the test bot_id '''
    clear_ns(f'persyn:{persyn_config.id.guid}:')

    for idx in [ltm.convo_prefix, ltm.summary_prefix]:
        try:
            ltm.redis.ft(idx).dropindex()
        except ltm.redis.exceptions.ResponseError as err:
            print(f"Couldn't drop index {idx}:", err)
