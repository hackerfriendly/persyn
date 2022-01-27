''' memory.py: long and short term memory by Elasticsearch. '''
import uuid

import urllib3
import elasticsearch

# Color logging
from color_logging import debug, info, warning, error, critical # pylint: disable=unused-import

# Time
from chrono import elapsed, get_cur_ts

# Disable SSL warnings for Elastic
urllib3.disable_warnings()

class LongTermMemory(): # pylint: disable=too-many-arguments
    ''' Wrapper class for Elasticsearch conversational memory. '''
    def __init__(
        self,
        url,
        auth_name,
        auth_key,
        convo_index,
        summary_index,
        conversation_interval=600, # 10 minutes
        verify_certs=True,
        timeout=30
    ):
        self.es = elasticsearch.Elasticsearch( # pylint: disable=invalid-name
            [url],
            http_auth=(auth_name, auth_key),
            verify_certs=verify_certs,
            timeout=timeout
        )
        self.index = {
            "convo": convo_index,
            "summary": summary_index
        }
        self.conversation_interval = conversation_interval

        for item in self.index.items():
            try:
                self.es.search(index=item[1], query={"match_all": {}}, size=1) # pylint: disable=unexpected-keyword-arg
            except elasticsearch.exceptions.NotFoundError:
                warning(f"Creating index {item[0]}")
                self.es.index(index=item[1], document={'@timestamp': get_cur_ts()}, refresh='true') # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

    def load_convo(self, channel, lines=16, summaries=3):
        '''
        Return a list of lines from the conversation index for this channel.
        If the conversation interval has elapsed, load summaries instead.
        '''
        convo_history = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['convo'],
            query={
                "term": {"channel.keyword": channel}
            },
            sort=[{"@timestamp":{"order":"desc"}}],
            size=lines
        )['hits']['hits']

        # Nothing in the channel
        if not convo_history:
            return []

        # Skip summaries if we hit max lines
        if len(convo_history) == lines:
            summaries = 0

        convo_id = convo_history[0]['_source']['convo_id']
        ret = self.load_summaries(channel, summaries)

        if self.time_to_move_on(convo_history[0]['_source']['@timestamp']):
            return ret

        for line in convo_history[::-1]:
            src = line['_source']
            if src['convo_id'] != convo_id:
                continue

            ret.append(f"{src['speaker']}: {src['msg']}")

        debug(f"load_convo(): {ret}")
        return ret

    def load_summaries(self, channel, summaries=3):
        '''
        Return a list of the most recent summaries for this channel.
        '''
        ret = []

        history = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['summary'],
            query={
                "term": {"channel.keyword": channel}
            },
            sort=[{"@timestamp":{"order":"desc"}}],
            size=summaries
        )['hits']['hits']

        for line in history[::-1]:
            src = line['_source']
            ret.append(src['summary'])

        debug(f"load_summaries(): {ret}")
        return ret

    def save_convo(self, channel, msg, speaker_id=None, speaker_name=None):
        '''
        Save a line of conversation to ElasticSearch.
        If the conversation interval has elapsed, start a new convo.
        Returns True if a new conversation was started, otherwise False.
        '''
        new_convo = True
        convo_id = uuid.uuid4()

        if not speaker_name:
            speaker_name = speaker_id

        cur_ts = get_cur_ts()
        last_message = self.get_last_message(channel)

        if last_message:
            prev_ts = last_message['_source']['@timestamp']

            if not self.time_to_move_on(prev_ts, cur_ts):
                new_convo = False
                convo_id = last_message['_source']['convo_id']
        else:
            prev_ts = cur_ts

        doc = {
            "@timestamp": cur_ts,
            "channel": channel,
            "speaker": speaker_name,
            "speaker_id": speaker_id,
            "msg": msg,
            "elapsed": elapsed(prev_ts, cur_ts),
            "convo_id": convo_id
        }
        _id = self.es.index(index=self.index['convo'], document=doc, refresh='true')["_id"] # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

        debug("doc:", _id)
        return new_convo

    def save_summary(self, channel, convo_id, summary):
        '''
        Save a conversation summary to ElasticSearch.
        '''
        doc = {
            "convo_id": convo_id,
            "summary": summary,
            "channel": channel,
            "@timestamp": get_cur_ts()
        }
        _id = self.es.index(index=self.index['summary'], document=doc, refresh='true')["_id"] # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        debug("doc:", _id)
        return True

    def get_last_message(self, channel):
        ''' Return the last message seen on this channel '''
        try:
            return self.es.search( # pylint: disable=unexpected-keyword-arg
                index=self.index['convo'],
                query={
                    "term": {"channel.keyword": channel}
                },
                sort=[{"@timestamp":{"order":"desc"}}],
                size=1
            )['hits']['hits'][0]
        except (KeyError, IndexError):
            return None

    def get_convo_by_id(self, convo_id):
        ''' Extract a full conversation by its convo_id. Returns a list of strings. '''
        history = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['convo'],
            query={
                "term": {"convo_id.keyword": convo_id}
            },
            sort=[{"@timestamp":{"order":"asc"}}],
            size=10000
        )['hits']['hits']

        ret = []
        for line in history:
            ret.append(f"{line['_source']['speaker']}: {line['_source']['msg']}")

        debug(f"get_convo_by_id({convo_id}):", ret)
        return ret

    def time_to_move_on(self, then, now=None):
        ''' Returns True if time elapsed between then and now is too long, otherwise False '''
        return elapsed(then, now or get_cur_ts()) > self.conversation_interval
