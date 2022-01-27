'''
memory (elasticsearch) tests
'''
import os
import datetime as dt

from time import sleep

from memory import LongTermMemory

prefix = os.environ['BOT_NAME'].lower()
now = dt.datetime.now().isoformat().replace(':','.').lower()

convo_index = f"{prefix}-test-conversations-{now}"
summary_index = f"{prefix}-test-summary-{now}"

def test_save_convo():
    ''' Make some test data '''
    ltm = LongTermMemory(
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        conversation_interval=1,  # New conversation every second
        verify_certs=False
    )
    # New convo
    assert ltm.save_convo("channel_a", "message_a", "speaker_id", "speaker_name") is True
    # Continued convo
    assert ltm.save_convo("channel_a", "message_b", "speaker_id", "speaker_name") is False
    # New convo again
    sleep(1.1)
    assert ltm.save_convo("channel_a", "message_c", "speaker_id", "speaker_name") is True

    # All new convos, speaker name / id are optional
    for i in range(2):
        assert ltm.save_convo(f"channel_loop_{i}", "message_loop_a", "speaker_id", "speaker_name") is True
        for j in range(3):
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_b{j}", speaker_id="speaker_id") is False
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_c{j}", speaker_name="speaker_name") is False
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_d{j}") is False

def test_fetch_convo():
    ''' Retrieve previously saved convo '''
    ltm = LongTermMemory(
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        verify_certs=False
    )
    assert len(ltm.load_convo('channel_loop_0')) == 10
    assert len(ltm.load_convo('channel_loop_0', lines=3)) == 3
    # First message (whole convo)
    assert ltm.load_convo('channel_loop_0')[0] == "speaker_name: message_loop_a"
    # Last message (most recent 1 line)
    assert ltm.load_convo('channel_loop_0', lines=1)[0] == "None: message_loop_d2"

    last_message = ltm.get_last_message('invalid_channel')
    assert not last_message

    last_message = ltm.get_last_message('channel_loop_1')
    assert last_message

    convo = ltm.get_convo_by_id(last_message['_source']['convo_id'])
    assert len(convo) == 10

def test_save_summaries():
    ''' Make some test data '''
    ltm = LongTermMemory(
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        verify_certs=False
    )
    assert ltm.save_summary("channel_a", "convo_id", "my_nice_summary") is True
    assert ltm.save_summary("channel_b", "convo_id_2", "my_other_nice_summary") is True
    assert ltm.save_summary("channel_b", "convo_id_3", "my_middle_nice_summary") is True
    assert ltm.save_summary("channel_b", "convo_id_4", "my_final_nice_summary") is True

def test_load_summaries():
    ''' Retrieve previously saved summaries '''
    ltm = LongTermMemory(
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        verify_certs=False
    )
    # zero lines returns empty list
    assert ltm.load_summaries('channel_a', 0) == [] # pylint: disable=use-implicit-booleaness-not-comparison
    # saved above
    assert ltm.load_summaries('channel_a') == ["my_nice_summary"]
    # correct order
    assert ltm.load_summaries('channel_b') == [
        "my_other_nice_summary",
        "my_middle_nice_summary",
        "my_final_nice_summary"
    ]

def test_fetch_convo_summarized():
    ''' Retrieve previously saved convo after an expired conversation_interval '''
    ltm = LongTermMemory(
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        conversation_interval=1,
        verify_certs=False
    )
    sleep(1.1)
    # contains only the summary
    assert ltm.load_convo('channel_a') == ["my_nice_summary"]

    # new convo
    assert ltm.save_convo("channel_a", "message_another", "speaker_id_2", "speaker_name_2") is True

    # contains the summary + new convo
    assert ltm.load_convo('channel_a') == ["my_nice_summary", "speaker_name_2: message_another"]
