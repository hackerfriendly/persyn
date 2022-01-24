'''
memory (elasticsearch) tests
'''
import os
import uuid
import datetime as dt

from time import sleep

from memory import LongTermMemory

prefix = os.environ['BOT_NAME'].lower()
now = dt.datetime.now().isoformat().replace(':','.').lower()

convo_index = f"{prefix}-test-conversations-{now}"
summary_index = f"{prefix}-test-summary-{now}"

ltm = LongTermMemory(
    url=os.environ["ELASTIC_URL"],
    auth_name=os.environ["BOT_NAME"],
    auth_key=os.environ.get("ELASTIC_KEY", None),
    convo_index=convo_index,
    summary_index=summary_index,
    conversation_interval=2,  # New conversation every second
    verify_certs=False,
)

def test_save_convo():
    ''' Make some test data '''
    # New convo
    assert ltm.save_convo(f"channel_a", f"message_a", "speaker_id", "speaker_name") == True
    sleep(1)
    # Continued convo
    assert ltm.save_convo(f"channel_a", f"message_b", "speaker_id", "speaker_name") == False
    sleep(2.1)
    # New convo again
    assert ltm.save_convo(f"channel_a", f"message_c", "speaker_id", "speaker_name") == True

    # All new convos, speaker name / id are optional
    for i in range(2):
        assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_a", "speaker_id", "speaker_name") == True
        sleep(1)
        for j in range(3):
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_b{j}", speaker_id="speaker_id") == False
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_c{j}", speaker_name="speaker_name") == False
            assert ltm.save_convo(f"channel_loop_{i}", f"message_loop_d{j}") == False

# def test_all_voices():
#     ''' Say hello in your native language '''
#     for voice in VOICES:
#         req = {
#             "text": f"Hello from {voice}",
#             "voice": voice
#         }
#         reply = requests.post(f'{os.environ["GTTS_SERVER_URL"]}/say/', params=req)
#         assert reply.ok
