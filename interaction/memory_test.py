'''
memory (elasticsearch) tests
'''
import os
import uuid
import datetime as dt

from time import sleep

import pytest

from memory import LongTermMemory
from chrono import elapsed, get_cur_ts

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
        verify_certs=False,
    )

    # New convo
    assert ltm.save_convo(f"channel_a", f"message_a", "speaker_id", "speaker_name") == True
    # Continued convo
    assert ltm.save_convo(f"channel_a", f"message_b", "speaker_id", "speaker_name") == False
    # New convo again
    sleep(1.1)
    assert ltm.save_convo(f"channel_a", f"message_c", "speaker_id", "speaker_name") == True

    # All new convos, speaker name / id are optional
    for i in range(2):
        assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_a", "speaker_id", "speaker_name") == True
        for j in range(3):
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_b{j}", speaker_id="speaker_id") == False
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_c{j}", speaker_name="speaker_name") == False
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_d{j}") == False

def test_fetch_convo():
    ''' Retrieve previously saved convo '''
    ltm = LongTermMemory(
        url=os.environ["ELASTIC_URL"],
        auth_name=os.environ["BOT_NAME"],
        auth_key=os.environ.get("ELASTIC_KEY", None),
        convo_index=convo_index,
        summary_index=summary_index,
        verify_certs=False,
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
