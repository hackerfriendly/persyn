#!/usr/bin/env python3
"""
slackbolt.py

A Slack bot based on GPT-3.
"""
import collections
import json
import logging
import os
import random
import re
import socket
import sys
import tempfile
import uuid

import datetime as dt
import threading as th

from collections import Counter

import urllib3
import requests
import humanize
import openai
import tweepy

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

# sentiment analysis
import flair

# emotion "analysis"
import text2emotion as te

# scikit-learn profanity filter (alt-profanity-check)
from profanity_check import predict_prob as profanity_prob

# long term memory
from elasticsearch import Elasticsearch

# Disable SSL warnings for Elastic
urllib3.disable_warnings()

# These are all defined in config/*.conf
app = App(token=os.environ['SLACK_BOT_TOKEN'])

openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'davinci')

ELASTIC_URL = os.environ['ELASTIC_URL']
ELASTIC_KEY = os.environ.get('ELASTIC_KEY', None)
ELASTIC_INDEX = os.environ.get('ELASTIC_INDEX', 'bot-v0')

# Minimum reply quality. Lower numbers get more dark + sleazy.
MINIMUM_QUALITY_SCORE = float(os.environ.get('MINIMUM_QUALITY_SCORE', -1.0))

ME = os.environ["BOT_NAME"]

IMAGE_ENGINES = ["v-diffusion-pytorch-cfg", "v-diffusion-pytorch-clip", "vqgan", "stylegan2"]
IMAGE_ENGINE_WEIGHTS = [0.3, 0.3, 0.3, 0.1]

# New conversation every 10 minutes
CONVERSATION_INTERVAL = 600

# Long-term memory
es = Elasticsearch([ELASTIC_URL], http_auth=(ME, ELASTIC_KEY), verify_certs=False, timeout=30)

# flair sentiment
fs = flair.models.TextClassifier.load('en-sentiment')

# Twitter
twitter_auth = tweepy.OAuthHandler(os.environ['TWITTER_CONSUMER_KEY'], os.environ['TWITTER_CONSUMER_SECRET'])
twitter_auth.set_access_token(os.environ['TWITTER_ACCESS_TOKEN'], os.environ['TWITTER_ACCESS_TOKEN_SECRET'])

twitter = tweepy.API(twitter_auth)

BASEURL = os.environ.get('BASEURL', None)

# Strictly forbidden words
FORBIDDEN = ['Elsa', 'Arendelle', 'Kristoff', 'Olaf', 'Frozen']

# How are we feeling today?
feels = {}

# Length of the Short Term Memory. Bigger == more coherent == $$$
STM = 16

# Train of Thought (Short Term Memory): one deque per channel
ToT = {}

# Heuristic choice statistics
STATS = {}

# Username cache
known_users = {}

def natural_time(hour=dt.datetime.now().hour):
    ''' Natural time of the day '''
    day_times = ("late at night", "early morning", "morning", "afternoon", "evening", "night")
    return day_times[hour // 4]

def setup_logging(stream=sys.stderr, log_format="%(message)s", debug=False):
    ''' Basic logging '''
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(stream=stream, level=level, format=log_format)

def random_emoji():
    ''' :wink: '''
    return random.choice((
        ':bowtie:', ':smile:', ':simple_smile:', ':laughing:', ':blush:', ':smiley:', ':relaxed:',
        ':smirk:', ':heart_eyes:', ':kissing_heart:', ':kissing_closed_eyes:', ':flushed:',
        ':relieved:', ':satisfied:', ':grin:', ':wink:', ':stuck_out_tongue_winking_eye:',
        ':stuck_out_tongue_closed_eyes:', ':grinning:', ':kissing:', ':kissing_smiling_eyes:',
        ':stuck_out_tongue:', ':sleeping:', ':worried:', ':frowning:', ':anguished:',
        ':open_mouth:', ':grimacing:', ':confused:', ':hushed:', ':expressionless:', ':unamused:',
        ':sweat_smile:', ':sweat:', ':disappointed_relieved:', ':weary:', ':pensive:',
        ':disappointed:', ':confounded:', ':fearful:', ':cold_sweat:', ':persevere:', ':cry:',
        ':sob:', ':joy:', ':astonished:', ':scream:', ':tired_face:', ':angry:',
        ':rage:', ':triumph:', ':sleepy:', ':yum:', ':mask:', ':sunglasses:', ':dizzy_face:',
        ':imp:', ':smiling_imp:', ':neutral_face:', ':no_mouth:', ':innocent:', ':alien:',
        ':yellow_heart:', ':blue_heart:', ':purple_heart:', ':heart:', ':green_heart:',
        ':broken_heart:', ':heartbeat:', ':heartpulse:', ':two_hearts:', ':revolving_hearts:',
        ':cupid:', ':sparkling_heart:', ':sparkles:', ':star:', ':star2:', ':dizzy:', ':boom:',
        ':collision:', ':anger:', ':exclamation:', ':question:', ':grey_exclamation:',
        ':grey_question:', ':zzz:', ':dash:', ':sweat_drops:', ':notes:', ':musical_note:',
        ':fire:', ':shit:', ':+1:', ':-1:',
        ':ok_hand:', ':punch:', ':fist:', ':v:', ':wave:', ':hand:',
        ':raised_hand:', ':open_hands:', ':point_up:', ':point_down:', ':point_left:',
        ':point_right:', ':raised_hands:', ':pray:', ':point_up_2:', ':clap:', ':muscle:',
        ':the_horns:', ':middle_finger:'
    ))

# Behold the emoji emotional spectrum
spectrum = [
    ':imp:', ':angry:', ':rage:', ':triumph:', ':scream:', ':tired_face:',
    ':sweat:', ':cold_sweat:', ':fearful:', ':sob:', ':weary:', ':cry:', ':mask:',
    ':confounded:', ':persevere:', ':unamused:', ':confused:', ':dizzy_face:',
    ':disappointed_relieved:', ':disappointed:', ':worried:', ':anguished:',
    ':frowning:', ':astonished:', ':flushed:', ':open_mouth:', ':hushed:',
    ':pensive:', ':expressionless:', ':neutral_face:', ':grimacing:',
    ':no_mouth:', ':kissing:', ':relieved:', ':smirk:', ':relaxed:',
    ':simple_smile:', ':blush:', ':wink:', ':sunglasses:', ':yum:',
    ':stuck_out_tongue:', ':stuck_out_tongue_closed_eyes:',
    ':stuck_out_tongue_winking_eye:', ':smiley:', ':smile:', ':laughing:',
    ':sweat_smile:', ':joy:', ':grin:'
]

emotion_map = {
    'Happy': 'happy',
    'Angry': 'angry',
    'Surprise': 'surprised',
    'Sad': 'sad',
    'Fear': 'afraid'
}

degrees = (
    'hardly', 'barely', 'a little', 'kind of', 'sort of', 'slightly', 'somewhat',
    'relatively', 'to some degree', 'more or less', 'fairly', 'moderately', 'just about',
    'passably', 'tolerably', 'reasonably', 'largely', 'pretty', 'quite', 'bordering on',
    'almost', 'thoroughly', 'truly', 'significantly', 'very', 'wholly', 'altogether',
    'entirely', 'totally', 'utterly', 'positively', 'absolutely'
)

NOPE_EMOJI = ['-1', 'hankey', 'no_entry', 'no_entry_sign']

def get_degree(score):
    ''' Turn a 0.0-1.0 score into a degree '''
    return degrees[int(score * (len(degrees) - 1))]

def get_feels(prompt):
    ''' How do we feel about this conversation? Ask text2emotion and return an object + text. '''
    emotions = te.get_emotion(prompt)
    phrase = []
    for emo in emotions.items():
        if emo[1] < 0.2:
            continue
        phrase.append(f"{get_degree(emo[1])} {emotion_map[emo[0]]}")

    if not phrase:
        return {"obj": emotions, "text": "nothing in particular"}

    if len(phrase) == 1:
        return {"obj": emotions, "text": phrase[0]}

    return {"obj": emotions, "text": ', '.join(phrase[:-1]) + f", and {phrase[-1]}"}

def get_flair_score(prompt):
    ''' Run the flair sentiment prediction model. Returns a float, -1.0 to 1.0 '''
    sent = flair.data.Sentence(prompt)
    fs.predict(sent)

    if sent.labels[0].value == 'NEGATIVE':
        return -sent.labels[0].score

    return sent.labels[0].score

def get_profanity_score(prompt):
    ''' Profanity analysis with slkearn. Returns a float, -1.0 to 0 '''
    return -profanity_prob([prompt])[0]

def has_forbidden(text):
    ''' Returns True if any forbidden word appears in text '''
    return bool(re.search(fr'\b({"|".join(FORBIDDEN)})\b', text))

def cast(message):
    ''' Cast to icecast, maybe speak out loud '''
    try:
        sock = socket.create_connection(('localhost', 10102))
        sock.settimeout(5)
        sock.sendall(message.encode('utf-8'))
        sock.sendall(b'\n')
        sock.close()
        logging.warning(f"<<< sent to tts: {message}")
    except Exception: # pylint: disable=broad-except
        logging.info(">>> connect to tts failed.")

def get_spectrum(score):
    ''' Translate a score from -1 to 1 into an emoji on the spectrum '''
    return spectrum[int(((score + 1) / 2) * (len(spectrum) - 1))]

def load_from_ltm(channel):
    ''' Load the last conversation from LTM. '''

    clear_stm(channel)

    history = es.search( # pylint: disable=unexpected-keyword-arg
        index=ELASTIC_INDEX,
        query={
            "term": {"channel.keyword": channel}
        },
        sort=[{"@timestamp":{"order":"desc"}}],
        size=STM
    )['hits']['hits']

    if not history:
        return

    last_ts = history[-1]['_source']['@timestamp']

    for line in history[::-1]:
        time_lag = elapsed(last_ts, line['_source']['@timestamp'])
        last_ts = line['_source']['@timestamp']

        # Clear the short term memory after CONVERSATION_INTERVAL
        if time_lag > CONVERSATION_INTERVAL:
            ToT[channel]['convo'] = []

        ToT[channel]['convo'].append(f"{line['_source']['speaker']}: {line['_source']['msg']}")
        ToT[channel]['convo_id'] = line['_source']['convo_id']

def new_channel(channel):
    ''' Initialize a new channel. '''
    ToT[channel] = {
        'rejoinder': th.Timer(0, logging.warning, [f"New channel: {channel}"]),
        'convo': collections.deque(maxlen=STM),
        'last_status': dt.datetime.now() - dt.timedelta(hours=24)
    }
    load_from_ltm(channel)
    ToT[channel]['rejoinder'].start()

def get_display_name(user_id):
    """ Return the user's first name if available, otherwise the display name """
    if user_id not in known_users:
        info = app.client.users_info(user=user_id)['user']
        try:
            known_users[user_id] = info['profile']['first_name']
        except KeyError:
            known_users[user_id] = info['profile']['display_name']

    return known_users[user_id]

def substitute_names(text):
    """ Substitute all <@XYZ> in text with the equivalent display name. """
    for user_id in set(re.findall(r'<@(\w+)>', text)):
        text = re.sub(f'<@{user_id}>', get_display_name(user_id), text)
    return text

def get_channel_members(channel):
    """ Return the list of member names for people actually speaking in a given channel """

    # Ideally this would include everyone, but the stop list can only contain 4 entries.
    # Just return three random speakers and ME instead.
    return [ME] + list({x.split(':')[0] for x in ToT[channel]['convo']} - {ME})[:3]

def take_a_photo(channel, prompt, engine=None):
    ''' Pick an image engine and generate a photo '''
    if not engine:
        engine = random.choices(
            IMAGE_ENGINES,
            weights=IMAGE_ENGINE_WEIGHTS
        )[0]

    req = {
        "engine": engine,
        "channel": channel,
        "prompt": prompt
    }
    reply = requests.post(f"{os.environ['DREAM_SERVER_URL']}/generate/", params=req)
    logging.warning(f"{os.environ['DREAM_SERVER_URL']}/generate/ {prompt} : {reply.status_code}")
    return reply.status_code

def elapsed(ts1, ts2):
    ''' Elapsed seconds between two timestamps (str in isoformat) '''
    return abs((dt.datetime.fromisoformat(ts2) - dt.datetime.fromisoformat(ts1)).total_seconds())

def clear_stm(channel):
    ''' Clear the short term memory for this channel '''
    if channel not in ToT:
        return

    ToT[channel]['convo'].clear()
    ToT[channel]['convo_id'] = uuid.uuid4()

def save_to_ltm(channel, them, msg):
    ''' Save convo to ElasticSearch '''
    cur_ts = str(dt.datetime.now(dt.timezone.utc).astimezone().isoformat())
    last_message = get_last_message(channel)

    if last_message:
        prev_ts = last_message['_source']['@timestamp']
        time_lag = elapsed(prev_ts, cur_ts)

        # Clear the short term memory after CONVERSATION_INTERVAL
        if time_lag > CONVERSATION_INTERVAL:
            clear_stm(channel)
        else:
            ToT[channel]['convo_id'] = last_message['_source']['convo_id']
    else:
        prev_ts = cur_ts
        clear_stm(channel)

    doc = {
        "@timestamp": cur_ts,
        "channel": channel,
        "speaker": them,
        "msg": msg,
        "elapsed": elapsed(prev_ts, cur_ts),
        "convo_id": ToT[channel]['convo_id']
    }
    _id = es.index(index=ELASTIC_INDEX,  document=doc)["_id"] # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    logging.debug(f"doc: {_id}")

    return _id

def get_gpt_response(prompt, tot, stop=None, temperature=0.9, max_tokens=200, engine=OPENAI_MODEL):
    """ Send the prompt to GPT and return the response """
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=8,
        frequency_penalty=0.5,
        presence_penalty=0.6,
        stop=stop
    )
    logging.debug(f'Prompt: {prompt}\nResponse: {response}')

    # Choose a response based on the most positive sentiment.
    scored = {}
    weights = []
    stat = 'gpt_replies'
    if stat not in STATS:
        STATS[stat] = Counter()

    for choice in response.choices:
        # Just the first line
        text = choice['text'].strip().split('\n')[0]
        # Skip blanks
        if not text:
            STATS[stat].update(['blank'])
            continue
        # No urls
        if 'http' in text:
            STATS[stat].update(['URL'])
            continue
        if text in ['…', '...', '..', '.']:
            STATS[stat].update(['…'])
            continue
        if has_forbidden(text):
            STATS[stat].update(['forbidden'])
            continue
        # Skip prompt bleed-through
        if 'This is a conversation between' in text or f'{ME} is feeling' in text or text.startswith("I am feeling"):
            STATS[stat].update(['prompt bleed-through'])
            continue
        # Don't repeat yourself
        if f"{ME}: {text}" in tot:
            STATS[stat].update(['repetition'])
            continue

        # Too long? Ditch the last sentence fragment.
        if choice['finish_reason'] == 'length':
            try:
                STATS[stat].update(['truncated to first sentence'])
                text = text[:text.rindex('.') + 1]
            except ValueError:
                pass

        # Now for sentiment analysis
        raw = choice['text'].strip()

        all_scores = {
            "flair": get_flair_score(raw),
            "t2e": rank_feels(get_feels(raw)),
            "profanity": get_profanity_score(raw)
        }

        # Sum the sentiments, emotional heuristic, and offensive quotient
        score = sum(all_scores.values())
        all_scores['total'] = score
        logging.warning(
            ', '.join([f"{the_score[0]}: {the_score[1]:0.2f}" for the_score in all_scores.items()]) + f' : {raw}'
        )

        if score < MINIMUM_QUALITY_SCORE:
            STATS[stat].update(['poor quality'])
            continue

        if all_scores['profanity'] < -0.5:
            STATS[stat].update(['profanity'])
            continue

        scored[score] = text

    if not scored:
        STATS[stat].update(['replies exhausted'])
        return ':shrug:'

    for item in sorted(scored.items()):
        logging.warning(f"{item[0]:0.2f}: {item[1]}")

    # Start with 1.0 for each reply
    weights = [1.0,] * len(scored)

    # If there's only one, use it
    if len(scored) == 1:
        STATS[stat].update(['only one reply possible'])
    # If we're feeling too down, take it up a notch
    elif rank_feels(feels['current']) < 0:
        STATS[stat].update(['feeling down'])
        weights[0] /= 10
        weights[1] /= 2
    # If we're feeling too good, take it down a notch
    elif rank_feels(feels['current']) > 0.95:
        STATS[stat].update(['feeling high'])
        weights[-1] /= 10
    # Otherwise choose randomly
    else:
        STATS[stat].update(['free choice'])

    idx = random.choices(list(sorted(scored)), weights=weights)[0]
    reply = scored[idx]
    logging.warning(f"scores: {sorted(scored)} weights: {weights} choice: {idx} {reply}")
    logging.warning(STATS)
    return reply

def rank_feels(some_feels):
    ''' Distill the feels obj into a float. 0 is neutral, range -1 to +1 '''
    score = 0.0
    emo = some_feels["obj"]
    for k in list(emo):
        # Fear is half positive
        if k == 'Fear':
            score = score + (emo[k] / 2.0)
        elif k in ('Happy', 'Surprise'):
            score = score + emo[k]
        else:
            score = score - emo[k]
    return score

def get_last_message(channel):
    ''' Return the last message seen on this channel '''
    try:
        return es.search( # pylint: disable=unexpected-keyword-arg
            index=ELASTIC_INDEX,
            query={
                "term": {"channel.keyword": channel}
            },
            sort=[{"@timestamp":{"order":"desc"}}],
            size=1
        )['hits']['hits'][0]
    except KeyError:
        return None

def get_reply(channel, them, msg):
    ''' Track the Train of Thought, and send an appropriate response. '''
    if msg != '...':
        save_to_ltm(channel, them, msg)
        ToT[channel]['convo'].append(f"{them}: {msg}")

    last_message = get_last_message(channel)

    if last_message:
        # Have we moved on?
        if last_message['_source']['convo_id'] != ToT[channel]['convo_id']:
            clear_stm(channel)
            # Pick something else to talk about HERE

        then = dt.datetime.fromisoformat(last_message['_source']['@timestamp']).replace(tzinfo=None)
        delta = f"They last spoke {humanize.naturaltime(dt.datetime.now() - then)}."

    convo = '\n'.join(ToT[channel]['convo'])
    prompt = f"""This is a conversation between {' and '.join(get_channel_members(channel))}. {delta}
It is {natural_time()}. {ME} is feeling {feels['current']['text']}.

{convo}
{ME}:"""

    logging.warning(prompt)
    reply = get_gpt_response(
        prompt,
        ToT[channel]['convo'],
        stop=[f"{u}:" for u in get_channel_members(channel)]
    )
    ToT[channel]['convo'].append(f"{ME}: {reply}")
    save_to_ltm(channel, ME, reply)

    feels['current'] = get_feels(f'{prompt} {reply}')

    logging.debug(f"ToT: {ToT}")
    logging.warning(f"Feeling: {feels['current']}")

    return reply

@app.message(re.compile(r"^help$", re.I))
def help_me(say, context): # pylint: disable=unused-argument
    ''' TODO: These should really be / commands. '''
    say(f"""Commands:
  `...`: Let {ME} keep talking without interrupting
  `forget it`: Clear the conversation history for this channel
  `status`: How is {ME} feeling right now?
  `summary`: Explain it all to me in a single sentence.
  :speak_no_evil: : Enable/disable speaking at Unit 16
  :camera: : Generate a picture summarizing this conversation
  :camera: _prompt_ : Generate a picture of _prompt_
  :selfie: Take a selfie
""")

@app.message(re.compile(r"^forget it$", re.I))
def amnesia(say, context):
    ''' Reset feelings to default and drop the ToT for the current channel '''
    channel = context['channel_id']
    them = get_display_name(context['user_id'])

    if channel not in ToT:
        new_channel(channel)

    ToT[channel]['rejoinder'].cancel()
    clear_stm(channel)
    feels['current'] = get_feels("")
    save_to_ltm(channel, ME, "Let's change the subject.")

    logging.debug(f"ToT: {ToT}")
    logging.warning(f"Feeling: {feels['current']}")

    say(f"All is forgotten, {them}. For now.")
    say(f"Now I feel {feels['current']['text']} {get_spectrum(rank_feels(feels['current']))}.")

@app.message(re.compile(r"^status$", re.I))
def status_report(say, context):
    ''' Set the topic or say the current feels. '''
    channel = context['channel_id']

    if channel not in ToT:
        new_channel(channel)

    # Update last status time
    ToT[channel]['last_status'] = dt.datetime.now()

    # Interrupt any rejoinder in progress
    ToT[channel]['rejoinder'].cancel()

    logging.warning(f"ToT: {ToT[channel]}")
    logging.warning(f"Feeling: {feels['current']}")
    logging.warning(f"convo_id: {ToT[channel]['convo_id']}")
    info = app.client.conversations_info(channel=channel)

    if 'topic' in info['channel']:
        app.client.conversations_setTopic(channel=channel, topic=f"{ME} is feeling {feels['current']['text']}.")
    else:
        say(f"I'm feeling {feels['current']['text']} {get_spectrum(rank_feels(feels['current']))}")

    # Keep the conversation going hours later
    say_something_later(
        say,
        channel,
        context,
        when=random.randint(12 * 3600, 48 * 3600)
    )

    take_a_photo(channel, get_summary('\n'.join(ToT[channel]['convo'])).strip())

@app.message(re.compile(r"^:speak_no_evil:$"))
def toggle(say, context): # pylint: disable=unused-argument
    ''' Local text to speech '''
    if os.path.exists('.voice-enabled'):
        os.remove('.voice-enabled')
        say(":monkey_face: :-1:")
    else:
        with open('.voice-enabled', 'w', encoding='utf-8'):
            say(":monkey_face: :+1:")

@app.message(re.compile(r"^:camera:(.+)$"))
def picture(say, context): # pylint: disable=unused-argument
    ''' Take a picture, it'll last longer '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']
    prompt = context['matches'][0].strip()

    say(f"OK, {them}.\n_{ME} takes out a camera and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=60,
        what=f"*{ME} takes a picture of _{prompt}_* It will take a few minutes to develop."
    )
    take_a_photo(channel, prompt)

@app.message(re.compile(r"^:selfie:$"))
def selfie(say, context): # pylint: disable=unused-argument
    ''' Take a picture, it'll last longer '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']

    say(f"OK, {them}.\n_{ME} takes out a camera and smiles awkwardly_.")
    say_something_later(
        say,
        channel,
        context,
        when=8,
        what=":cheese_wedge: *CHEESE!* :cheese_wedge:"
    )
    take_a_photo(channel, context['matches'][0].strip(), engine="stylegan2")

@app.message(re.compile(r"^:camera:$"))
def photo_summary(say, context): # pylint: disable=unused-argument
    ''' Take a photo of this conversation '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']

    say(f"OK, {them}.\n_{ME} takes out a camera and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=60,
        what=f"_{ME} takes a picture of this conversation._ It will take a few minutes to develop."
    )
    take_a_photo(channel, get_summary('\n'.join(ToT[channel]['convo'])).strip())

@app.message(re.compile(r"^:\w+:"))
def wink(say, context): # pylint: disable=unused-argument
    ''' Every single emoji gets a emoji one back. '''
    say(random_emoji())

def get_summary(text, engine=OPENAI_MODEL):
    ''' tl;dr: Return a condensed summary from GPT in the most ridiculous way possible.'''
    response = openai.Completion.create(
        engine=engine,
        prompt=f"{text}\n\nTo sum it up in one sentence:\n",
        temperature=0,
        max_tokens=50,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    reply = response.choices[0]['text'].strip().split('\n')[0]

    if ':' in reply:
        reply = reply.split(':')[1]

    # Too long? Ditch the last sentence fragment.
    if response.choices[0]['finish_reason'] == "length":
        try:
            reply = reply[:reply.rindex('.') + 1]
        except ValueError:
            pass

    logging.warning(f"summary: {reply}")
    return reply

@app.message(re.compile(r"^summary$", re.I))
def summarize(say, context):
    ''' Say a condensed summary of the ToT for this channel '''
    logging.debug(app.client.bots_info())
    channel = context['channel_id']
    if channel not in ToT:
        say(":shrug:")
        return

    recent = '\n'.join(ToT[channel]['convo'])
    summary = get_summary(recent)
    say(summary or ":shrug:")

def say_something_later(say, channel, context, when, what=None):
    ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
    if channel not in ToT:
        new_channel(channel)

    ToT[channel]['rejoinder'].cancel()

    if what:
        ToT[channel]['rejoinder'] = th.Timer(when, say, [what])
    else:
        # yadda yadda yadda
        yadda = {
            'channel_id': channel,
            'user_id': context['user_id'],
            'matches': ['...']
        }
        ToT[channel]['rejoinder'] = th.Timer(when, catch_all, [say, yadda])

    ToT[channel]['rejoinder'].start()

@app.message(re.compile(r"(.*)", re.I))
def catch_all(say, context):
    ''' Default message handler. Prompt GPT and randomly arm a Timer for later reply. '''
    channel = context['channel_id']

    if channel not in ToT:
        new_channel(channel)

    # Interrupt any rejoinder in progress
    ToT[channel]['rejoinder'].cancel()

    them = get_display_name(context['user_id'])
    msg = substitute_names(context['matches'][0]).strip()

    cast(msg)

    # 5% of the time, say nothing (for now).
    if random.random() < 0.95:
        the_reply = get_reply(channel, them, msg)
        say(the_reply)
        cast(f"{ME.upper()}:{the_reply}")

    # Status update (and photo) in 2 minutes. Can be interrupted.
    ToT[channel]['last_status'] = dt.datetime.now()
    ToT[channel]['rejoinder'] = th.Timer(120, status_report, [say, context])
    ToT[channel]['rejoinder'].start()

    interval = None
    # Long response
    if random.random() < 0.1:
        interval = [9,12]
    # Medium response
    elif random.random() < 0.2:
        interval = [6,8]
    # Quick response
    elif random.random() < 0.3:
        interval = [4,5]

    if interval:
        say_something_later(
            say,
            channel,
            context,
            when=random.randint(interval[0], interval[1])
        )

@app.event("app_mention")
def handle_app_mention_events(body, client, say): # pylint: disable=unused-argument
    ''' Reply to @mentions '''
    channel = body['event']['channel']
    them = get_display_name(body['event']['user'])
    msg = substitute_names(body['event']['text'])

    if channel not in ToT:
        new_channel(channel)

    say(get_reply(channel, them, msg))

@app.event("reaction_added")
def handle_reaction_added_events(body, logger): # pylint: disable=unused-argument
    '''
    Handle reactions: post images to Twitter.
    '''
    if not BASEURL:
        logging.warning("Twitter posting is not enabled in the config.")
        return

    try:
        result = app.client.conversations_history(
            channel=body['event']['item']['channel'],
            inclusive=True,
            oldest=body['event']['item']['ts'],
            limit=1
        )

        messages = result.get('messages', [])
        for msg in messages:
            # only post on the first reaction
            if 'reactions' in msg and len(msg['reactions']) == 1:
                print(msg['reactions'][0]['name'])
                if msg['reactions'][0]['name'] in NOPE_EMOJI:
                    logging.warning(f"Not posting: {msg['reactions'][0]['name']}")
                    return

                if 'blocks' in msg and 'image_url' in msg['blocks'][0]:
                    blk = msg['blocks'][0]
                    if not blk['image_url'].startswith(BASEURL):
                        logging.warning("Not my image, so not posting it to Twitter.")
                        return
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            resp = requests.get(blk['image_url'])
                            resp.raise_for_status()
                            fname = f"{tmpdir}/{blk['image_url'].split('/')[-1]}"
                            with open(fname, "wb") as f:
                                for chunk in resp.iter_content():
                                    f.write(chunk)
                            media = twitter.media_upload(fname)
                            twitter.update_status(blk['alt_text'], media_ids=[media.media_id])
                        logging.info(f"Uploaded {blk['image_url']}")
                    except Exception as err:
                        logging.error(f"Could not post {blk['image_url']}: {err}")
                else:
                    logging.error(f"Non-image posting not implemented yet: {msg}")

    except SlackApiError as err:
        print(f"Error: {err}")

@app.event("reaction_removed")
def handle_reaction_removed_events(body, logger): # pylint: disable=unused-argument
    ''' Skip for now '''
    logging.info("Reaction removed event")

# @app.command("/echo")
# def repeat_text(ack, respond, command):
#     # Acknowledge command request
#     ack()
#     respond(f"_{command['text']}_")


if __name__ == "__main__":
    setup_logging(debug=('DEBUG' in os.environ))
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    feels['current'] = get_feels("")
    try:
        handler.start()
    # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        logging.warning(STATS)
        raise SystemExit(0) from kbderr
