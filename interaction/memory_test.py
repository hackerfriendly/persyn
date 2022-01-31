'''
memory (elasticsearch) tests
'''
import os
import datetime as dt
import uuid

from time import sleep

from memory import LongTermMemory

prefix = os.environ['BOT_NAME'].lower()
now = dt.datetime.now().isoformat().replace(':','.').lower()

convo_index = f"{prefix}-test-conversations-{now}"
summary_index = f"{prefix}-test-summary-{now}"
entity_index = f"{prefix}-test-entity-{now}"
relation_index = f"{prefix}-test-relation-{now}"

def test_save_convo():
    ''' Make some test data '''
    ltm = LongTermMemory(
        bot_name=os.environ["BOT_NAME"],
        bot_id=os.environ["BOT_ID"],
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        entity_index=entity_index,
        relation_index=relation_index,
        conversation_interval=1,  # New conversation every second
        verify_certs=False
    )
    # New convo
    assert ltm.save_convo("my_service", "channel_a", "message_a", "speaker_name", "speaker_id") is True
    # Continued convo
    assert ltm.save_convo("my_service", "channel_a", "message_b", "speaker_name", "speaker_id") is False
    # New convo again
    sleep(1.1)
    assert ltm.save_convo("my_service", "channel_a", "message_c", "speaker_name", "speaker_id") is True

    # All new convos, speaker name / id are optional
    for i in range(2):
        assert ltm.save_convo("my_service", f"channel_loop_{i}", "message_loop_a", "speaker_name", "speaker_id") is True
        for j in range(3):
            assert ltm.save_convo(
                "my_service",
                f"channel_loop_{i}",
                f"message_loop_b{j}",
                speaker_id="speaker_id") is False
            assert ltm.save_convo(
                "my_service",
                f"channel_loop_{i}",
                f"message_loop_c{j}",
                speaker_name="speaker_name") is False
            assert ltm.save_convo("my_service", f"channel_loop_{i}", f"message_loop_d{j}") is False

def test_fetch_convo():
    ''' Retrieve previously saved convo '''
    ltm = LongTermMemory(
        bot_name=os.environ["BOT_NAME"],
        bot_id=os.environ["BOT_ID"],
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        entity_index=entity_index,
        relation_index=relation_index,
        verify_certs=False
    )
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
    ltm = LongTermMemory(
        bot_name=os.environ["BOT_NAME"],
        bot_id=os.environ["BOT_ID"],
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        entity_index=entity_index,
        relation_index=relation_index,
        verify_certs=False
    )
    assert ltm.save_summary("my_service", "channel_a", "convo_id", "my_nice_summary") is True
    assert ltm.save_summary("my_service", "channel_b", "convo_id_2", "my_other_nice_summary") is True
    assert ltm.save_summary("my_service", "channel_b", "convo_id_3", "my_middle_nice_summary") is True
    assert ltm.save_summary("my_service", "channel_b", "convo_id_4", "my_final_nice_summary") is True

def test_load_summaries():
    ''' Retrieve previously saved summaries '''
    ltm = LongTermMemory(
        bot_name=os.environ["BOT_NAME"],
        bot_id=os.environ["BOT_ID"],
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        entity_index=entity_index,
        relation_index=relation_index,
        verify_certs=False
    )
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

def test_fetch_convo_summarized():
    ''' Retrieve previously saved convo after an expired conversation_interval '''
    ltm = LongTermMemory(
        bot_name=os.environ["BOT_NAME"],
        bot_id=os.environ["BOT_ID"],
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        entity_index=entity_index,
        relation_index=relation_index,
        conversation_interval=1,
        verify_certs=False
    )
    sleep(1.1)
    # contains only the summary
    assert ltm.load_convo("my_service", "channel_a") == ["my_nice_summary"]

    # new convo
    assert ltm.save_convo("my_service", "channel_a", "message_another", "speaker_name_2", "speaker_id") is True

    # contains the summary + new convo
    assert ltm.load_convo("my_service", "channel_a") == ["my_nice_summary", "speaker_name_2: message_another"]

def test_entities():
    ''' Exercise entity generation and lookup '''
    ltm = LongTermMemory(
        bot_name=os.environ["BOT_NAME"],
        bot_id=os.environ["BOT_ID"],
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        entity_index=entity_index,
        relation_index=relation_index,
        verify_certs=False
    )

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

def test_short_ids():
    ''' shortuuid support '''
    ltm = LongTermMemory(
        bot_name=os.environ["BOT_NAME"],
        bot_id=os.environ["BOT_ID"],
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        entity_index=entity_index,
        relation_index=relation_index,
        verify_certs=False
    )

    random_uuid = uuid.uuid4()
    entity_id = ltm.uuid_to_entity(random_uuid)
    assert str(random_uuid) == ltm.entity_to_uuid(entity_id)

    entity_id = ltm.uuid_to_entity(str(random_uuid))
    assert str(random_uuid) == ltm.entity_to_uuid(entity_id)
