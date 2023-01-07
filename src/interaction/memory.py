''' memory.py: long and short term memory by Elasticsearch. '''
# pylint: disable=invalid-name
import uuid

import elasticsearch
import shortuuid as su

# Time
from interaction.chrono import elapsed, get_cur_ts

# Relationship graphs
from interaction.relationships import get_relationship_graph, ranked_matches, graph_to_json

# Color logging
from utils.color_logging import ColorLog

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

    def save(self, service, channel, msg, speaker_name, speaker_id, verb=None):
        ''' Save to stm and ltm. Clears stm if it expired. Returns the current convo_id. '''
        if self.stm.expired(service, channel):
            self.stm.clear(service, channel)

        return self.stm.append(
            service,
            channel,
            self.ltm.save_convo(
                service,
                channel,
                msg,
                speaker_name,
                speaker_id,
                self.stm.convo_id(service, channel),
                verb
            )
        )

    def summary(self, service, channel, summary, keywords=None):
        ''' Save a summary. Clears stm. '''
        if not summary:
            return False

        if keywords is None:
            keywords = []

        convo_id = self.stm.convo_id(service, channel)
        self.stm.clear(service, channel)
        self.ltm.save_summary(service, channel, convo_id, summary, keywords)
        return True

    def expired(self, service, channel):
        ''' True if this conversation has expired, else False '''
        return self.stm.expired(service, channel)

    def forget(self, service, channel):
        ''' What were we talking about? '''
        self.stm.clear(service, channel)
        return True

    def add_goal(self, service, channel, goal):
        ''' Add a goal to this channel. '''
        return self.stm.add_goal(service, channel, goal)

    def get_goals(self, service, channel):
        ''' Return the temparary goals for this channel, if any. '''
        return self.stm.get_goals(service, channel)

    def lookup_summaries(self, service, channel, search, size=1):
        ''' Oh right. '''
        return self.ltm.lookup_summaries(service, channel, search, size)

    def judge(self, service, channel, topic, opinion, speaker_id=None):
        ''' Judge not, lest ye be judged '''
        log.warning(f"👨‍⚖️ judging {topic}")
        return self.ltm.save_opinion(service, channel, topic, opinion, speaker_id)

    def opine(self, service, channel, topic, speaker_id=None, size=10):
        ''' Everyone's got an opinion '''
        log.warning(f"🧷 opinion on {topic}")
        return self.ltm.lookup_opinion(topic, service, channel, speaker_id, size)

    def dialog(self, service, channel):
        ''' Return the dialog from stm (if any) '''
        convo = self.stm.fetch(service, channel)
        if convo:
            return [
                f"{line['speaker']}: {line['msg']}" for line in convo
                if 'verb' not in line or line['verb'] == 'dialog'
            ]
        return []

    def convo(self, service, channel):
        '''
        Return the entire convo from stm (if any).

        Result is a list of "speaker (verb): msg" strings.
        '''
        convo = self.stm.fetch(service, channel)
        if not convo:
            return []

        ret = []
        for line in convo:
            if 'verb' not in line or line['verb'] in ['dialog', None]:
                ret.append(f"{line['speaker']}: {line['msg']}")
            else:
                ret.append(f"{line['speaker']} {line['verb']}: {line['msg']}")

        return ret

    def summaries(self, service, channel, size=3):
        ''' Return the summary text from ltm (if any) '''
        return [s['_source']['summary'] for s in self.ltm.lookup_summaries(service, channel, None, size=size)]

    def lts(self, service, channel):
        ''' Return the timestamp of the last message from this channel (if any) '''
        return self.ltm.get_last_timestamp(service, channel)

    def find_related_convos(self, service, channel, size=1, edge_bias=0.5):
        '''
        Find conversations related to the current convo using ES score and graph analysis.

        Returns a ranked list of graph hits.
        '''
        # No convo? Nothing to match.
        convo = self.convo(service, channel)
        if not convo:
            return []

        convo_text = ' '.join(convo)

        # No relationships? Nothing to match.
        hits = self.ltm.lookup_relationships(service, channel, convo_text, size)
        if not hits:
            log.info("👨‍👩‍👧‍👦 find_related_convos():", "No hits, nothing to match.")
            return []

        G = get_relationship_graph(convo_text)

        ranked = ranked_matches(G, hits, edge_bias=edge_bias)
        log.info("👨‍👩‍👧‍👦 find_related_convos():", f"{len(ranked)} matches")
        return ranked

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
        self.convo[service][channel]['opinions'] = []
        self.convo[service][channel]['goals'] = []

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

    def append(self, service, channel, convo_obj):
        '''
        Append a convo object to a channel. If the current channel expired, clear it first.
        Returns the convo_id.
        '''
        if not self.exists(service, channel):
            self.create(service, channel)
        if self.expired(service, channel):
            self.clear(service, channel)

        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'].append(convo_obj)

        return self.convo_id(service, channel)

    def add_bias(self, service, channel, line):
        '''
        Append a short-term opinion to a channel. Returns the convo_id.
        '''
        self.convo[service][channel]['opinions'].append(line)
        return self.convo_id(service, channel)

    def get_bias(self, service, channel):
        '''
        Return all short-term opinions for a channel.
        '''
        return self.convo[service][channel]['opinions']

    def add_goal(self, service, channel, goal):
        '''
        Append a short-term goal to a channel. Returns the convo_id.
        '''
        if not self.exists(service, channel):
            self.create(service, channel)

        if goal not in self.convo[service][channel]['goals']:
            self.convo[service][channel]['goals'].append(goal)

        # five max
        if len(self.convo[service][channel]['goals']) > 5:
            self.convo[service][channel]['goals'] = self.convo[service][channel]['goals'][:5]

        return self.convo[service][channel]['goals']

    def get_goals(self, service, channel):
        '''
        Return all short-term goals for a channel.
        '''
        if not self.exists(service, channel):
            return []
        return self.convo[service][channel]['goals']

    def fetch(self, service, channel):
        ''' Fetch the current convo, if any '''
        if not self.exists(service, channel):
            return []
        return self.convo[service][channel]['convo']

    def opinions(self, service, channel):
        ''' Fetch the current opinions expressed in convo, if any '''
        if not self.exists(service, channel):
            return []
        return self.convo[service][channel]['opinions']

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
            basic_auth=(auth_name, auth_key),
            verify_certs=verify_certs,
            request_timeout=timeout
        )

        self.index = {
            "convo": f"{self.index_prefix}-conversations-{self.version}",
            "summary": f"{self.index_prefix}-summaries-{self.version}",
            "entity": f"{self.index_prefix}-entities-{self.version}",
            "relationship": f"{self.index_prefix}-relationships-{self.version}",
            "opinion": f"{self.index_prefix}-opinions-{self.version}",
            "belief": f"{self.index_prefix}-beliefs-{self.version}"
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
                # if item[0] == "relationhip":
                #     self.save_relationship(self.uuid_to_entity(bot_id), "has_name", self.bot_name)

    def get_last_timestamp(self, service, channel):
        '''
        Get the timestamp of the last message, or the current ts if there is none.
        '''
        try:
            return self.get_last_message(service, channel)['_source']['@timestamp']
        except (KeyError, TypeError):
            return get_cur_ts()

    def save_convo(
        self,
        service,
        channel,
        msg,
        speaker_name=None,
        speaker_id=None,
        convo_id=None,
        verb=None,
        refresh=False
    ):
        '''
        Save a line of conversation to ElasticSearch. Returns the convo doc.
        '''
        prev_ts = self.get_last_timestamp(service, channel)

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
            "verb": verb,
            "convo_id": convo_id,
            "elapsed": elapsed(prev_ts, cur_ts)
        }
        _id = self.es.index( # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            index=self.index['convo'],
            document=doc,
            refresh='true' if refresh else 'false'
        )["_id"]

        doc['_id'] = _id
        doc['_index'] = self.index['convo']

        log.debug("doc:", doc)
        return doc

    # TODO: set refresh=False after separate summary thread is implemented.
    def save_summary(self, service, channel, convo_id, summary, keywords=None, refresh=True):
        '''
        Save a conversation summary to ElasticSearch.
        '''
        if keywords is None:
            keywords = []

        cur_ts = get_cur_ts()
        doc = {
            "convo_id": convo_id,
            "summary": summary,
            "service": service,
            "channel": channel,
            "@timestamp": cur_ts,
            "keywords": keywords
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
        '''
        Extract a full conversation by its convo_id.
        Returns a list convo objects.
        '''
        ret = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['convo'],
            query={
                "term": {"convo_id.keyword": convo_id}
            },
            sort=[{"@timestamp":{"order":"asc"}}],
            size=10000
        )['hits']['hits']

        log.debug(f"get_convo_by_id({convo_id}):", ret)
        return ret

    def lookup_summaries(self, service, channel, search=None, size=3):
        '''
        Return a list of summaries matching the search term for this channel.
        '''
        # TODO: match speaker id HERE when cross-channel entity merging is working
        query = {
            "bool": {
                "must": [
                    {"match": {"service.keyword": service}},
                    {"match": {"channel.keyword": channel}},
                ]
            }
        }

        if search:
            query['bool']['must'].append({"match": {"summary": {"query": search}}})

        history = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['summary'],
            query=query,
            sort=[{"@timestamp":{"order":"desc"}}],
            size=size
        )['hits']['hits']

        ret = []
        for hit in history[::-1]:
            ret.append(hit)

        log.debug(f"lookup_summaries(): {ret}")
        return ret

    def lookup_relationships(self, service, channel, search=None, size=3):
        '''
        Return a list of convo graphs matching the search term for this channel.
        '''
        # TODO: match speaker id HERE when cross-channel entity merging is working
        query = {
            "bool": {
                "must": [
                    {"match": {"service.keyword": service}},
                    {"match": {"channel.keyword": channel}},
                ]
            }
        }

        if search:
            query['bool']['must'].append({"match": {"convo": {"query": search}}})

        history = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['relationship'],
            query=query,
            size=size
        )['hits']['hits']

        ret = []
        for hit in history[::-1]:
            ret.append(hit)

        log.debug(f"lookup_relationships(): {ret}")
        return ret

    def save_relationship_graph(self, service, channel, convo_id, text, include_archetypes=True):
        ''' Save a relationship graph '''
        doc = {
            '@timestamp': get_cur_ts(),
            'service': service,
            'channel': channel,
            'convo_id': convo_id,
            'graph': graph_to_json(
                get_relationship_graph(text, include_archetypes=include_archetypes)
            ),
            'convo': text,
            'refresh': False
        }
        rep = self.save_relationship(**doc)
        if rep['result'] != 'created':
            log.critical("∑ Could not save relationship:", rep)
        else:
            log.info("∑ relationship saved.")

    @staticmethod
    def entity_key(service, channel, name):
        ''' Unique string for each service + channel + name '''
        return f"{str(service).strip()}|{str(channel).strip()}|{str(name).strip()}"

    @staticmethod
    def uuid_to_entity(the_uuid):
        ''' Return the equivalent short ID (str) for a uuid (str) '''
        return str(su.encode(uuid.UUID(str(the_uuid))))

    @staticmethod
    def entity_to_uuid(entity_id):
        ''' Return the equivalent UUID (str) for a short ID (str) '''
        return str(su.decode(entity_id))

    def name_to_entity(self, service, channel, name):
        ''' One distinct short UUID per bot_id + service + channel + name '''
        return self.uuid_to_entity(uuid.uuid5(self.bot_id, self.entity_key(service, channel, name)))

    def save_entity(self, service, channel, speaker_name, speaker_id=None, entity_id=None, refresh=True):
        '''
        If an entity is new, save it to Elasticscarch.
        Returns the entity_id and the elapsed time since the entity was first stored.
        '''
        if not speaker_id:
            speaker_id = speaker_name

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
            document=doc,
            refresh='true' if refresh else 'false'
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

    def save_opinion(self, service, channel, topic, opinion, speaker_id=None, refresh=True):
        '''
        Save an opinion to Elasticscarch.
        '''
        if not speaker_id:
            speaker_id = self.bot_id

        doc = {
            "service": service,
            "channel": channel,
            "topic": topic.lower(),
            "opinion": opinion,
            "speaker_id": speaker_id,
            "@timestamp": get_cur_ts()
        }
        self.es.index( # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            index=self.index['opinion'],
            document=doc,
            refresh='true' if refresh else 'false'
        )
        # return something here?

    def lookup_opinion(self, topic, service=None, channel=None, speaker_id=None, size=10):
        ''' Look up an opinion in Elasticsearch. '''
        query = {
            "bool": {
                "must": [
                    {"match": {"topic.keyword": topic.lower()}}
                ]
            }
        }

        if service:
            query["bool"]["must"].append({"match": {"service.keyword": service}})

        if channel:
            query["bool"]["must"].append({"match": {"channel.keyword": channel}})

        if speaker_id:
            query["bool"]["must"].append({"match": {"speaker_id.keyword": speaker_id}})

        ret = []
        for opinion in self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['opinion'],
            query=query,
            size=size
        )['hits']['hits']:
            ret.append(opinion["_source"]["opinion"])

        return ret

    def save_relationship(self, service, channel, refresh=True, **kwargs):
        '''
        Save a relationship to Elasticearch.
        '''
        if 'source_id' in kwargs and not all(['rel' in kwargs, 'target_id' in kwargs]):
            log.critical('👩‍👧‍👦 source_id requires rel and target_id')
            return None

        if 'convo_id' in kwargs and 'graph' not in kwargs:
            log.critical('👩‍👧‍👦 convo_id requires graph')
            return None

        cur_ts = get_cur_ts()
        doc = {
            "@timestamp": cur_ts,
            "service": service,
            "channel": channel
        }

        for term, val in kwargs.items():
            doc[term] = val

        ret = self.es.index( # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            index=self.index['relationship'],
            document=doc,
            refresh='true' if refresh else 'false'
        )
        return ret

    def lookup_relationship(self, service, channel, size=10, **kwargs):
        ''' Look up a relationship in Elasticsearch. '''
        ret = []

        query = {
            "bool": {
                "must": [
                    {"match": {"service.keyword": service}},
                    {"match": {"channel.keyword": channel}}
                ]
            }
        }

        for term, val in kwargs.items():
            query['bool']['must'].append({"match": {f'{term}.keyword': val}})

        log.debug(f"👨‍👩‍👧 query: {query}")

        ret = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['relationship'],
            query=query,
            sort=[{"@timestamp":{"order":"desc"}}],
            size=size
        )['hits']['hits']

        log.debug(f"👨‍👩‍👧 return: {ret}")
        return ret
