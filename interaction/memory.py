''' memory.py: long and short term memory by Elasticsearch. '''
import uuid

import elasticsearch
import shortuuid as su

# Time
from chrono import elapsed, get_cur_ts

# Color logging
from color_logging import ColorLog

log = ColorLog()

class ShortTermMemory():
    ''' Wrapper class for in-process short term conversational memory. '''
    def __init__(self, conversation_interval):
        self.convo = {}
        self.conversation_interval = conversation_interval

    def exists(self, service, channel):
        ''' True if a channel already exists, else False '''
        return service in self.convo and channel in self.convo[service]

    def create(self, service, channel):
        ''' Create a new empty channel. Erases existing channel if any. '''
        if not self.exists(service, channel):
            if service not in self.convo:
                self.convo[service] = {}
            if channel not in self.convo[service]:
                self.convo[service][channel] = {}
        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'] = []

    def clear(self, service, channel):
        ''' Clear a channel. '''
        if not self.exists(service, channel):
            return
        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'].clear()

    def expired(self, service, channel):
        ''' True if time elapsed since last update is > conversation_interval, else False '''
        if not self.exists(service, channel):
            return True
        return elapsed(self.convo[service][channel]['ts'], get_cur_ts()) > self.conversation_interval

    def append(self, service, channel, line):
        ''' Append a line to a channel. If the current channel expired, clear it first. '''
        if not self.exists(service, channel):
            self.create(service, channel)
        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'].append(line)

    def fetch(self, service, channel):
        ''' Fetch the current convo '''
        if not self.exists(service, channel):
            self.create(service, channel)
        return self.convo[service][channel]['convo']

    def last(self, service, channel):
        ''' Fetch the last message from this convo (if any) '''
        if not self.exists(service, channel):
            return None

        last = self.fetch(service, channel)
        if last:
            return last[-1]

        return None

class LongTermMemory(): # pylint: disable=too-many-arguments
    ''' Wrapper class for Elasticsearch conversational memory. '''
    def __init__(
        self,
        bot_name,
        bot_id,
        url,
        auth_name,
        auth_key,
        convo_index,
        summary_index,
        entity_index,
        relation_index,
        conversation_interval=600, # 10 minutes
        verify_certs=True,
        timeout=30
    ):
        self.bot_name = bot_name
        self.bot_id = uuid.UUID(bot_id)
        self.bot_entity_id = self.uuid_to_entity(bot_id)
        self.es = elasticsearch.Elasticsearch( # pylint: disable=invalid-name
            [url],
            http_auth=(auth_name, auth_key),
            verify_certs=verify_certs,
            timeout=timeout
        )
        self.index = {
            "convo": convo_index,
            "summary": summary_index,
            "entity": entity_index,
            "relation": relation_index
        }
        self.conversation_interval = conversation_interval
        self.stm=ShortTermMemory(conversation_interval)

        for item in self.index.items():
            try:
                self.es.search(index=item[1], query={"match_all": {}}, size=1) # pylint: disable=unexpected-keyword-arg
            except elasticsearch.exceptions.NotFoundError:
                log.warning(f"Creating index {item[0]}")
                self.es.index(index=item[1], document={'@timestamp': get_cur_ts()}, refresh='true') # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
                if item[0] == "entity":
                    self.save_entity(
                        service="self",
                        channel="bootstrap",
                        speaker_name=self.bot_name,
                        speaker_id=self.bot_id,
                        entity_id=self.bot_entity_id
                    )
                # if item[0] == "relation":
                #     self.save_relationship(self.uuid_to_entity(bot_id), "has_name", self.bot_name)

    def load_convo(self, service, channel, lines=16, summaries=3):
        '''
        Return a list of lines from the conversation index for this channel.
        If the conversation interval has elapsed, load summaries instead.
        '''
        convo_history = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['convo'],
            query={
                "bool": {
                    "must": [
                        {"match": {"service.keyword": service}},
                        {"match": {"channel.keyword": channel}}
                    ]
                }
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
        ret = self.load_summaries(service, channel, summaries)

        if self.time_to_move_on(convo_history[0]['_source']['@timestamp']):
            return ret

        for line in convo_history[::-1]:
            src = line['_source']
            if src['convo_id'] != convo_id:
                continue

            ret.append(f"{src['speaker']}: {src['msg']}")

        log.debug(f"load_convo(): {ret}")
        return ret

    def load_summaries(self, service, channel, summaries=3):
        '''
        Return a list of the most recent summaries for this channel.
        '''
        ret = []

        history = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['summary'],
            query={
                "bool": {
                    "must": [
                        {"match": {"service.keyword": service}},
                        {"match": {"channel.keyword": channel}}
                    ]
                }
            },
            sort=[{"@timestamp":{"order":"desc"}}],
            size=summaries
        )['hits']['hits']

        for line in history[::-1]:
            src = line['_source']
            ret.append(src['summary'])

        log.debug(f"load_summaries(): {ret}")
        return ret

    def save_convo(self, service, channel, msg, speaker_name=None, speaker_id=None):
        '''
        Save a line of conversation to ElasticSearch.
        If the conversation interval has elapsed, start a new convo.
        Returns True if a new conversation was started, otherwise False.
        '''
        new_convo = True
        convo_id = su.encode(uuid.uuid4())

        if speaker_id == self.bot_id:
            entity_id = self.bot_entity_id
        else:
            entity_id, _ = self.save_entity(service, channel, speaker_name, speaker_id)

        cur_ts = get_cur_ts()
        last_message = self.get_last_message(service, channel)

        if last_message:
            prev_ts = last_message['_source']['@timestamp']

            if not self.time_to_move_on(prev_ts, cur_ts):
                new_convo = False
                convo_id = last_message['_source']['convo_id']
        else:
            prev_ts = cur_ts

        doc = {
            "@timestamp": cur_ts,
            "service": service,
            "channel": channel,
            "speaker": speaker_name,
            "speaker_id": speaker_id,
            "entity_id": entity_id,
            "msg": msg,
            "elapsed": elapsed(prev_ts, cur_ts),
            "convo_id": convo_id
        }
        _id = self.es.index(index=self.index['convo'], document=doc, refresh='false')["_id"] # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

        log.debug("doc:", _id)
        return new_convo

    def save_summary(self, service, channel, convo_id, summary):
        '''
        Save a conversation summary to ElasticSearch.
        '''
        doc = {
            "convo_id": convo_id,
            "summary": summary,
            "service": service,
            "channel": channel,
            "@timestamp": get_cur_ts()
        }
        _id = self.es.index(index=self.index['summary'], document=doc, refresh='false')["_id"] # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        log.debug("doc:", _id)
        return True

    def get_last_message(self, service, channel):
        ''' Return the last message seen on this channel '''
        if not self.stm.expired(service, channel):
            return self.stm.last(service, channel)
# rjf
        try:
            return self.es.search( # pylint: disable=unexpected-keyword-arg
                index=self.index['convo'],
                query={
                    "bool": {
                        "must": [
                            {"match": {"service.keyword": service}},
                            {"match": {"channel.keyword": channel}}
                        ]
                    }
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

        log.debug(f"get_convo_by_id({convo_id}):", ret)
        return ret

    def time_to_move_on(self, then, now=None):
        ''' Returns True if time elapsed between then and now is too long, otherwise False '''
        return elapsed(then, now or get_cur_ts()) > self.conversation_interval

    @staticmethod
    def entity_key(service, channel, name):
        ''' Unique string for each service + channel + name '''
        return f"{str(service).strip()}|{str(channel).strip()}|{str(name).strip()}"

    @staticmethod
    def uuid_to_entity(the_uuid):
        ''' Return the equivalent short ID (str) for a uuid '''
        return str(su.encode(uuid.UUID(str(the_uuid))))

    @staticmethod
    def entity_to_uuid(entity_id):
        ''' Return the equivalent UUID (str) for a uuid '''
        return str(su.decode(entity_id))

    def name_to_entity(self, service, channel, name):
        ''' One distinct short UUID per bot_id + service + channel + name '''
        return self.uuid_to_entity(uuid.uuid5(self.bot_id, self.entity_key(service, channel, name)))

    def save_entity(self, service, channel, speaker_name, speaker_id=None, entity_id=None):
        '''
        If an entity is new, save it to Elasticscarch.
        Returns the entity_id and the elapsed time since the entity was first stored.
        '''
        if not speaker_id:
            speaker_name = speaker_id

        if not entity_id:
            entity_id = self.name_to_entity(service, channel, speaker_id)
        entity = self.lookup_entity(entity_id)

        if entity:
            return entity_id, elapsed(entity['@timestamp'], get_cur_ts())

        doc = {
            "service": service,
            "channel": channel,
            "speaker_name": speaker_name,
            "speaker_id": speaker_id,
            "entity_id": entity_id,
            "@timestamp": get_cur_ts()
        }
        self.es.index( # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            index=self.index['entity'],
            document=doc
        )
        return entity_id, 0

    def lookup_entity(self, entity_id):
        ''' Look up an entity_id in Elasticsearch. '''
        entity = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['entity'],
            query={
                "term": {"entity_id.keyword": entity_id}
            },
            size=1
        )['hits']['hits']

        if entity:
            return entity[0]['_source']

        return []

    def entity_to_name(self, entity_id):
        ''' If the entity_id exists, return its name. '''
        entity = self.lookup_entity(entity_id)
        if entity:
            return entity['speaker_name']
        return []
