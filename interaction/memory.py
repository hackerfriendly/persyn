''' memory.py: long and short term memory by Elasticsearch. '''
import uuid

import elasticsearch
import shortuuid as su

# Time
from chrono import elapsed, get_cur_ts

# Color logging
from color_logging import ColorLog

log = ColorLog()

class Recall(): # pylint: disable=too-many-arguments
    ''' Total Recall: stm + ltm. '''
    def __init__(
        self,
        bot_name,
        bot_id,
        url,
        auth_name,
        auth_key,
        index_prefix=None,
        conversation_interval=600, # 10 minutes
        verify_certs=True,
        version="v0",
        timeout=30
    ):
        self.bot_name = bot_name
        self.bot_id = uuid.UUID(bot_id)

        self.stm = ShortTermMemory(conversation_interval)
        self.ltm = LongTermMemory(
            bot_name,
            bot_id,
            url,
            auth_name,
            auth_key,
            index_prefix=index_prefix,
            version=version,
            verify_certs=verify_certs,
            timeout=timeout
        )

    def load(self, service, channel, summaries=3):
        ''' Return some summaries and the contents of the stm '''
        the_summaries = self.ltm.load_summaries(service, channel, summaries)

        if self.stm.expired(service, channel):
            return the_summaries

        return the_summaries + self.stm.fetch(service, channel)

    def save(self, service, channel, msg, speaker_name, speaker_id):
        ''' Save to stm and ltm. Clears stm if it expired. '''
        if self.stm.expired(service, channel):
            self.stm.clear(service, channel)

        convo_id = self.stm.append(service, channel, f"{speaker_name}: {msg}")
        self.ltm.save_convo(service, channel, msg, speaker_name, speaker_id, convo_id)
        return True

    def summary(self, service, channel, summary):
        ''' Save a summary. Clears stm. '''
        convo_id = self.stm.convo_id(service, channel)
        self.stm.clear(service, channel)
        self.ltm.save_summary(service, channel, convo_id, summary)
        return True

    def expired(self, service, channel):
        ''' True if this conversation has expired, else False '''
        return self.stm.expired(service, channel)

class ShortTermMemory():
    ''' Wrapper class for in-process short term conversational memory. '''
    def __init__(self, conversation_interval):
        self.convo = {}
        self.conversation_interval = conversation_interval

    def _new(self, service, channel):
        ''' Immediately initialize a new channel without sanity checking '''
        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'] = []
        self.convo[service][channel]['id'] = su.encode(uuid.uuid4())

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
        self._new(service, channel)

    def clear(self, service, channel):
        ''' Clear a channel. '''
        if not self.exists(service, channel):
            return
        self._new(service, channel)

    def expired(self, service, channel):
        ''' True if time elapsed since last update is > conversation_interval, else False '''
        if not self.exists(service, channel):
            return True
        return elapsed(self.convo[service][channel]['ts'], get_cur_ts()) > self.conversation_interval

    def append(self, service, channel, line):
        '''
        Append a line to a channel. If the current channel expired, clear it first.
        Returns the convo_id.
        '''
        if not self.exists(service, channel):
            self.create(service, channel)
        if self.expired(service, channel):
            self.clear(service, channel)

        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'].append(line)

        return self.convo_id(service, channel)

    def fetch(self, service, channel):
        ''' Fetch the current convo, if any '''
        if not self.exists(service, channel):
            return None
        return self.convo[service][channel]['convo']

    def convo_id(self, service, channel):
        ''' Return the convo id, if any '''
        if not self.exists(service, channel):
            return None
        return self.convo[service][channel]['id']

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
        index_prefix=None,
        verify_certs=True,
        timeout=30,
        version="v0"
    ):
        self.bot_name = bot_name
        self.bot_id = uuid.UUID(bot_id)
        self.bot_entity_id = self.uuid_to_entity(bot_id)
        self.index_prefix = index_prefix or bot_name.lower()
        self.version = version

        self.es = elasticsearch.Elasticsearch( # pylint: disable=invalid-name
            [url],
            http_auth=(auth_name, auth_key),
            verify_certs=verify_certs,
            timeout=timeout
        )

        self.index = {
            "convo": f"{self.index_prefix}-conversations-{self.version}",
            "summary": f"{self.index_prefix}-summaries-{self.version}",
            "entity": f"{self.index_prefix}-entities-{self.version}",
            "relation": f"{self.index_prefix}-relationships-{self.version}"
        }

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

    def save_convo(
        self,
        service,
        channel,
        msg,
        speaker_name=None,
        speaker_id=None,
        convo_id=None,
        prev_ts=get_cur_ts(),
        refresh=True
    ):
        '''
        Save a line of conversation to ElasticSearch. Returns the convo_id and current timestamp.
        '''
        if not convo_id:
            convo_id = su.encode(uuid.uuid4())

        # Save speaker entity
        if speaker_id == self.bot_id:
            entity_id = self.bot_entity_id
        else:
            entity_id, _ = self.save_entity(service, channel, speaker_name, speaker_id)

        cur_ts = get_cur_ts()
        doc = {
            "@timestamp": cur_ts,
            "service": service,
            "channel": channel,
            "speaker": speaker_name,
            "speaker_id": speaker_id,
            "entity_id": entity_id,
            "msg": msg,
            "convo_id": convo_id,
            "elapsed": elapsed(prev_ts, cur_ts)
        }
        _id = self.es.index( # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            index=self.index['convo'],
            document=doc,
            refresh='true' if refresh else 'false'
        )["_id"]

        log.debug("doc:", _id)
        return convo_id, cur_ts

    def save_summary(self, service, channel, convo_id, summary, refresh=True):
        '''
        Save a conversation summary to ElasticSearch.
        '''
        cur_ts = get_cur_ts()
        doc = {
            "convo_id": convo_id,
            "summary": summary,
            "service": service,
            "channel": channel,
            "@timestamp": cur_ts
        }
        _id = self.es.index( # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            index=self.index['summary'],
            document=doc,
            refresh='true' if refresh else 'false'
        )["_id"]

        log.debug("doc:", _id)
        return cur_ts

    def get_last_message(self, service, channel):
        ''' Return the last message seen on this channel '''
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
