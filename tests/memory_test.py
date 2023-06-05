'''
memory (redis) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, no-member
import uuid

from time import sleep

import ulid
import pytest

from src.persyn.interaction.memory import Recall, Person, Thing

# Bot config
from src.persyn.utils.config import load_config

# from utils.color_logging import log

persyn_config = load_config()

def test_basics():
    ''' Exercise the short term memory '''
    recall = Recall(persyn_config, conversation_interval=2)

    service = 'service1'
    channel = 'channel1'

    # start fresh
    assert not recall.get_last_message(service, channel)
    ret = recall.save_convo_line(service, channel, 'my_message', 'me')
    assert ret
    assert not recall.expired(service, channel)
    assert recall.get_last_message(service, channel)

    # service and channel are distinct
    assert not recall.get_last_message("another", channel)
    assert not recall.get_last_message(service, "different")

    # add some lines
    recall.save_convo_line(service, channel, "foo", "me", convo_id=ret.convo_id)
    assert recall.get_last_message(service, channel).msg == "foo"

    recall.save_convo_line(service, channel, "bar", "them", convo_id=ret.convo_id)
    assert recall.get_last_message(service, channel).msg == "bar"

    assert recall.convo(service, channel) == ["me: my_message", "me: foo", "them: bar"]

    convo_id = recall.convo_id(service, channel)
    assert convo_id

    # convo change
    sleep(2)
    assert recall.expired(service, channel)
    # expiration clears id
    assert recall.convo_id(service, channel) != convo_id


def test_short_ids():
    ''' ulid support '''
    recall = Recall(persyn_config, conversation_interval=10)

    random_uuid = uuid.uuid4()
    entity_id = recall.uuid_to_entity(random_uuid)

    # should be shorter than uuid
    assert len(str(entity_id)) < len(str(random_uuid))

    assert random_uuid == recall.entity_to_uuid(entity_id)

    assert recall.uuid_to_entity(random_uuid) == recall.uuid_to_entity(recall.entity_to_uuid(entity_id))
    assert recall.uuid_to_entity(uuid.uuid4()) != recall.uuid_to_entity(recall.entity_to_uuid(entity_id))

    random_uuid = uuid.uuid4()
    entity_id = recall.uuid_to_entity(random_uuid)
    assert recall.entity_to_uuid(entity_id) == recall.entity_to_uuid(recall.uuid_to_entity(random_uuid))


# def test_entities():
#     ''' Exercise entity generation and lookup '''
#     service = "my_service"
#     channel = "channel_a"
#     speaker_name = "test_name"
#     speaker_id = "test_id"

#     # This is computed using persyn_config.id.guid. If it changes, this value needs updating.
#     eid = recall.name_to_entity_id(service, channel, speaker_id)
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
#     # assert recall.save_entity(service, channel, speaker_name, speaker_id)[1] == 0
#     # assert recall.save_entity(service, channel, speaker_name)[1] == 0
#     # sleep(1.1)
#     # assert recall.save_entity(service, channel, speaker_name, speaker_id)[1] > 1
#     # assert recall.save_entity(service, channel, speaker_name, speaker_id)[1] < 8

#     # Should match
#     assert recall.entity_id_to_name(eid) == speaker_name

#     # All fields
#     doc = recall.lookup_entity_id(eid)
#     assert doc['service'] == service
#     assert doc['channel'] == channel
#     assert doc['speaker_name'] == speaker_name
#     assert doc['speaker_id'] == speaker_id

def cleanup():
    ''' Delete everything with the test bot_id '''
    recall = Recall(persyn_config)

    clear_ns(f'persyn:{persyn_config.id.guid}:')

    for idx in [recall.convo_prefix, recall.summary_prefix]:
        try:
            recall.redis.ft(idx).dropindex()
        except recall.redis.exceptions.ResponseError as err:
            print(f"Couldn't drop index {idx}:", err)

def test_generate_convos(service="my_service", channel_a="channel_a", channel_b="channel_b"):
    ''' Make some test data '''
    recall = Recall(persyn_config, conversation_interval=600)

    # New convo
    doc1 = recall.save_convo_line(
        service=service,
        channel=channel_a,
        msg="message_a",
        speaker_id="speaker_id",
        speaker_name="speaker_name",
    )

    # Continued convo
    doc2 = recall.save_convo_line(
        service=service,
        channel=channel_a,
        convo_id=str(doc1.convo_id),
        msg="message_b",
        speaker_id="speaker_id",
        speaker_name="speaker_name",
    )
    assert doc1.convo_id == doc2.convo_id
    assert recall.entity_id_to_epoch(doc1.pk) != recall.entity_id_to_epoch(doc2.pk)

    # New convo
    new_convo_id = str(ulid.ULID())
    doc3 = recall.save_convo_line(
        service=service,
        channel=channel_a,
        convo_id=new_convo_id,
        msg="message_c",
        speaker_id="speaker_id",
        speaker_name="speaker_name",
    )
    assert doc3.convo_id == new_convo_id

    # All new convos, speaker name / id are optional

    for i in range(2):
        doc4 = recall.save_convo_line(
            service=service,
            channel=f"channel_loop_{i}",
            msg="message_loop_a",
            speaker_id="speaker_id",
            speaker_name="speaker_name",
        )
        assert doc4

        for j in range(3):
            doc5 = recall.save_convo_line(
                service=service,
                channel=f"channel_loop_{i}",
                convo_id=str(doc4.convo_id),
                msg=f"message_loop_b{j}",
                speaker_id="speaker_id",
                speaker_name="speaker_name",
            )
            assert doc4.convo_id == doc5.convo_id

            doc6 = recall.save_convo_line(
                service=service,
                channel=f"channel_loop_{i}",
                convo_id=str(doc4.convo_id),
                msg=f"message_loop_c{j}",
                speaker_id="speaker_id",
                speaker_name="speaker_name",
            )
            assert doc5.convo_id == doc6.convo_id
            assert recall.entity_id_to_epoch(doc6.pk) - recall.entity_id_to_epoch(doc5.pk) < 2.0

            sleep(0.1)

            # Assert refresh on the last msg so we can fetch later
            doc7 = recall.save_convo_line(
                service=service,
                channel=f"channel_loop_{i}",
                convo_id=str(doc4.convo_id),
                msg=f"message_loop_d{j}",
                speaker_id="speaker_id",
                speaker_name="speaker_name",
            )
            assert doc4.convo_id == doc7.convo_id
            assert recall.entity_id_to_epoch(doc7.pk) - recall.entity_id_to_epoch(doc4.pk) < 15.0

    # Save some summaries too
    assert recall.save_summary(service, channel_a, str(ulid.ULID()), "my_nice_summary")
    assert recall.save_summary(service, channel_b, str(ulid.ULID()), "my_other_nice_summary")
    assert recall.save_summary(service, channel_b, str(ulid.ULID()), "my_middle_nice_summary")
    assert recall.save_summary(service, channel_b, str(ulid.ULID()), "my_final_nice_summary")


def test_fetch_convo():
    ''' Retrieve previously saved convo '''
    recall = Recall(persyn_config, conversation_interval=600)

    last_message = recall.get_last_message("my_service", "invalid_channel")
    assert not last_message

    last_message = recall.get_last_message("another_service", "channel_loop_1")
    assert not last_message

    last_message = recall.get_last_message("my_service", "channel_loop_1")
    assert last_message

    convo = recall.get_convo_by_id(last_message.convo_id)
    assert len(convo) == 10

def test_summaries():
    ''' Retrieve previously saved summaries '''
    recall = Recall(persyn_config)

    # zero lines returns empty list
    assert recall.summaries("my_service", "channel_a", None, 0) == [] # pylint: disable=use-implicit-booleaness-not-comparison
    # saved above
    assert recall.summaries("my_service", "channel_a", None, size=3, raw=False) == ["my_nice_summary"]
    # equivalent
    assert [
        s.summary
        for s in recall.summaries("my_service", "channel_a", None, size=3, raw=True)
    ] == ["my_nice_summary"]

    # correct order
    assert recall.summaries("my_service", "channel_b", None, size=3) == [
        "my_other_nice_summary",
        "my_middle_nice_summary",
        "my_final_nice_summary"
    ]

def test_recall():
    ''' Autogenerate summaries '''

    recall = Recall(persyn_config, conversation_interval=1)

    # Must match generate_convos()
    service = "my_service"
    channel = "channel_a"

    # contains only the summary
    convo = recall.convo(service, channel)
    assert recall.summaries(service, channel) == ["my_nice_summary"]
    assert not convo

    # new convo
    convo_id = recall.save_convo_line(
        service,
        channel,
        msg="message_another",
        speaker_name="speaker_name_1",
        speaker_id="speaker_id"
    ).convo_id

    # contains the summary + new convo
    c = recall.convo(service, channel, raw=True)
    convo = recall.convo(service, channel)

    assert recall.summaries(service, channel) == ["my_nice_summary"]
    assert c[0].convo_id == convo_id
    assert c[0].speaker_name == "speaker_name_1"
    assert c[0].msg == "message_another"
    assert convo == ["speaker_name_1: message_another"]

    # same convo
    assert recall.save_convo_line(
        service,
        channel,
        msg="message_yet_another",
        speaker_name="speaker_name_2",
        speaker_id="speaker_id",
        convo_id=convo_id
    )

    # contains the summary + new convo
    c = recall.convo(service, channel, raw=True)
    convo = recall.convo(service, channel)
    assert recall.summaries(service, channel) == ["my_nice_summary"]
    assert (c[0].speaker_name, c[0].msg) == ("speaker_name_1", "message_another")
    assert (c[1].speaker_name, c[1].msg) == ("speaker_name_2", "message_yet_another")
    assert convo == ["speaker_name_1: message_another", "speaker_name_2: message_yet_another"]

    # summarize
    assert recall.save_summary(service, channel, c[0].convo_id, "this_is_another_summary")

    # time passes...
    sleep(2)

    # expired
    assert recall.expired(service, channel)

    # only summaries
    s = recall.summaries(service, channel)
    c = recall.convo(service, channel, verb='dialog')
    assert (s, c) == (
        ["my_nice_summary", "this_is_another_summary"],
        []
    )

def test_memory_selection():
    ''' Find appropriate memories using cosine similarity '''

    recall = Recall(persyn_config, conversation_interval=600)

    service = "memory_selection"

    # new convo
    assert recall.save_convo_line(service, "channel_a", "Why did the cow become a painter?", "Anna", "anna_id")
    assert recall.save_convo_line(service, "channel_a", "No idea.", "Rob", "rob_id")
    assert recall.save_convo_line(service, "channel_a", "Because it had a real moo-sterpiece in mind!", "Anna", "anna_id")
    assert recall.save_convo_line(service, "channel_a", "Udderly terrible.", "Rob", "rob_id")

    assert recall.save_convo_line(service, "channel_b", "Why was the cat sitting on the computer?", "Anna", "anna_id")
    assert recall.save_convo_line(service, "channel_b", "I give up.", "Rob", "rob_id")
    assert recall.save_convo_line(service, "channel_b", "Because it wanted to keep an eye on the mouse!", "Anna", "anna_id")
    assert recall.save_convo_line(service, "channel_b", "🙄", "Rob", "rob_id")

    # not found on channel_a
    assert len(recall.find_related_convos(service, 'channel_a', 'cat sitting', size=5, threshold=0.2, any_convo=False)) == 0
    # found if any_convo == True
    assert len(recall.find_related_convos(service, 'channel_a', 'cat sitting', size=5, threshold=0.16, any_convo=True)) == 1
    # found on channel_b
    assert len(recall.find_related_convos(service, 'channel_b', 'cat sitting', size=5, threshold=0.16)) == 1
    # synonym found
    assert len(recall.find_related_convos(service, 'channel_a', 'awful', size=5, threshold=0.2, any_convo=False)) == 1
    assert recall.find_related_convos(service, 'channel_a', 'awful', size=5, threshold=0.2, any_convo=False)[0].msg == 'Udderly terrible.'
    # not found on channel_b
    assert len(recall.find_related_convos(service, 'channel_b', 'awful', size=5, threshold=0.2, any_convo=False)) == 0


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
    assert recall.list_goals(service, channel) == []

    # add a goal
    recall.add_goal(service, channel, goal)
    assert recall.list_goals(service, channel) == [goal]

    # adding it again has no effect
    recall.add_goal(service, channel, goal)
    assert recall.list_goals(service, channel) == [goal]

    # multiple goals
    recall.add_goal(service, channel, goal2)
    assert recall.list_goals(service, channel) == [goal, goal2]

    # achieve one
    recall.achieve_goal(service, channel, goal)
    assert recall.list_goals(service, channel) == [goal2]

    # Goals on other channels have no impact
    recall.add_goal(service, channel2, goal)
    assert recall.list_goals(service, channel2) == [goal]
    assert recall.list_goals(service, channel) == [goal2]

    recall.add_goal(service, channel2, goal2)
    assert recall.list_goals(service, channel) == [goal2]
    assert recall.list_goals(service, channel2) == [goal, goal2]

    recall.achieve_goal(service, channel2, goal2)
    assert recall.list_goals(service, channel) == [goal2]
    assert recall.list_goals(service, channel2) == [goal]

    # achieving a nonexistent goal has no effect
    recall.achieve_goal(service, channel2, goal2)
    assert recall.list_goals(service, channel2) == [goal]

    recall.achieve_goal(service, channel2, goal)
    assert recall.list_goals(service, channel2) == []


def test_news():
    ''' Store news urls '''
    recall = Recall(persyn_config)

    opts = {
        "service": "my_service",
        "channel": "my_channel",
        "url": "http://persyn.io",
    }

    assert recall.have_read(**opts) is False
    assert recall.add_news(title="The Persyn Codebase", **opts)
    assert recall.have_read(**opts) is True


def test_kg():
    ''' Neo4j tests '''
    recall = Recall(persyn_config)

    recall.triples_to_kg([("This", "isOnly", "aTest")])
    assert len(list(recall.fetch_all_nodes())) == 2
    assert recall.find_node(name='aTest').first().name == 'aTest'
    assert len(list(recall.find_node(name='aTest', node_type='person'))) == 0

    with pytest.raises(RuntimeError):
        assert recall.find_node(name='This', node_type='invalid')

    with pytest.raises(Person.DoesNotExist):
        recall.find_node(name='This', node_type='person').first()

    node = recall.find_node(name='This', node_type='thing').first()
    assert node.name == 'This'

    recall.delete_all_nodes(confirm=True)

    with pytest.raises(Thing.DoesNotExist):
        assert recall.find_node(name='This', node_type='thing').first()


def clear_ns(ns, chunk_size=5000):
    ''' Clear a namespace '''
    recall = Recall(persyn_config)

    # persyn:[guid]: == 44 chars
    if not ns or len(ns) != 44:
        return False

    cursor = '0'
    while cursor != 0:
        cursor, keys = recall.redis.scan(cursor=cursor, match=f"{ns}*", count=chunk_size)
        if keys:
            recall.redis.delete(*keys)
    return True

def test_cleanup():
    ''' Clean it all up at the end '''
    cleanup()
