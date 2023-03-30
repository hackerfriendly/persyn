''' memory.py: long and short term memory by Elasticsearch. '''
# pylint: disable=invalid-name, no-name-in-module, abstract-method, no-member
import re
import uuid
import logging

import elasticsearch
import shortuuid as su

from neomodel import DateTimeProperty, StringProperty, UniqueIdProperty, FloatProperty, IntegerProperty, RelationshipTo, StructuredRel, Q
from neomodel import config as neomodel_config
from neomodel import db as neomodel_db
from neomodel.contrib import SemiStructuredNode

# Time
from interaction.chrono import elapsed, get_cur_ts

# Relationship graphs
from interaction.relationships import Relationships

# Color logging
from utils.color_logging import ColorLog

log = ColorLog()

# Silence Elasticsearch transport
logging.getLogger('elastic_transport.transport').setLevel(logging.CRITICAL)

# Neomodel (Neo4j) graph classes
class Entity(SemiStructuredNode):
    ''' Superclass for everything '''
    name = StringProperty(required=True)
    bot_id = StringProperty(required=True)

class PredicateRel(StructuredRel):
    ''' Predicate relationship: s P o '''
    weight = IntegerProperty(default=0)
    verb = StringProperty(required=True)

class Person(Entity):
    ''' Something you can have a conversation with '''
    created = DateTimeProperty(default_now=True)
    last_contact = DateTimeProperty(default_now=True)
    entity_id = UniqueIdProperty()
    speaker_id = UniqueIdProperty()
    link = RelationshipTo('Entity', 'LINK', model=PredicateRel)

class Thing(Entity):
    ''' Any old thing '''
    created = DateTimeProperty(default_now=True)
    link = RelationshipTo('Entity', 'LINK', model=PredicateRel)

class Human(Person):
    ''' Flesh and blood '''
    last_contact = DateTimeProperty(default_now=True)
    trust = FloatProperty(default=0.0)

class Bot(Person):
    ''' Silicon and electricity '''
    service = StringProperty(required=True)
    channel = StringProperty(required=True)
    trust = FloatProperty(default=0.0)


class Recall():
    ''' Total Recall: stm + ltm. '''
    def __init__(self, persyn_config, version=None, conversation_interval=None):
        self.bot_name = persyn_config.id.name
        self.bot_id = uuid.UUID(persyn_config.id.guid)

        self.stm = ShortTermMemory(persyn_config, conversation_interval)
        self.ltm = LongTermMemory(persyn_config, version)

    def save(self, service, channel, msg, speaker_name, speaker_id, verb=None, convo_id=None):
        '''
        Save to stm and ltm. Clears stm if it expired. Returns the current convo_id.

        Specify a different convo_id to override the value in ltm.
        '''
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
                convo_id or self.stm.convo_id(service, channel),
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
        return self.ltm.add_goal(service, channel, goal)

    def achieve_goal(self, service, channel, goal):
        ''' Achieve a goal from this channel. '''
        return self.ltm.achieve_goal(service, channel, goal)

    def get_goals(self, service, channel, goal=None, achieved=None, size=10):
        ''' Return the goals for this channel, if any. '''
        return self.ltm.get_goals(service, channel, goal, achieved, size)

    def list_goals(self, service, channel, achieved=False, size=10):
        ''' Return a simple list of goals for this channel, if any. '''
        return self.ltm.list_goals(service, channel, achieved, size)

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

    def convo(self, service, channel, feels=False):
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
            elif feels or line['verb'] != ['feels']:
                ret.append(f"{line['speaker']} {line['verb']}: {line['msg']}")

        return ret

    def feels(self, convo_id):
        '''
        Return the last known feels for this channel.
        '''
        convo = self.ltm.get_convo_by_id(convo_id)
        log.debug("📢", convo)
        for doc in convo[::-1]:
            if doc['_source']['verb'] == 'feels':
                return doc['_source']['msg']

        return "nothing in particular"

    def summaries(self, service, channel, size=3):
        ''' Return the summary text from ltm (if any) '''
        return [s['_source']['summary'] for s in self.ltm.lookup_summaries(service, channel, None, size=size)]

    def lts(self, service, channel):
        ''' Return the timestamp of the last message from this channel (if any) '''
        return self.ltm.get_last_timestamp(service, channel)

class ShortTermMemory():
    ''' Wrapper class for in-process short term conversational memory. '''
    def __init__(self, persyn_config, conversation_interval=None):
        self.convo = {}
        self.persyn_config = persyn_config
        self.conversation_interval = conversation_interval or persyn_config.memory.conversation_interval

    def _new(self, service, channel):
        ''' Immediately initialize a new channel without sanity checking '''
        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'] = []
        self.convo[service][channel]['id'] = su.encode(uuid.uuid4())
        self.convo[service][channel]['opinions'] = []

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

    def add_bias(self, service, channel, opinion):
        '''
        Append a short-term opinion to a channel if it's not already there. Returns the current opinions.
        '''
        if not self.exists(service, channel):
            self.create(service, channel)
        if opinion not in self.convo[service][channel]['opinions']:
            self.convo[service][channel]['opinions'].append(opinion)
        return self.convo[service][channel]['opinions']

    def get_bias(self, service, channel):
        '''
        Return all short-term opinions for a channel.
        '''
        if not self.exists(service, channel):
            self.create(service, channel)
        return self.convo[service][channel]['opinions']

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
    def __init__(self, persyn_config, version=None):
        self.relationships = Relationships(persyn_config)
        self.persyn_config = persyn_config
        self.bot_name = persyn_config.id.name
        self.bot_id = uuid.UUID(persyn_config.id.guid)
        self.bot_entity_id = self.uuid_to_entity(persyn_config.id.guid)
        self.index_prefix = persyn_config.memory.elastic.index_prefix
        self.version = version or persyn_config.memory.elastic.version

        self.es = elasticsearch.Elasticsearch( # pylint: disable=invalid-name
            [persyn_config.memory.elastic.url],
            basic_auth=(persyn_config.memory.elastic.user, persyn_config.memory.elastic.key),
            verify_certs=persyn_config.memory.elastic.verify_certs,
            request_timeout=persyn_config.memory.elastic.timeout
        )

        self.index = {
            "convo": f"{self.index_prefix}-conversations-{self.version}",
            "summary": f"{self.index_prefix}-summaries-{self.version}",
            "entity": f"{self.index_prefix}-entities-{self.version}",
            "relationship": f"{self.index_prefix}-relationships-{self.version}",
            "opinion": f"{self.index_prefix}-opinions-{self.version}",
            "goal": f"{self.index_prefix}-goals-{self.version}",
            "news": f"{self.index_prefix}-news-{self.version}",
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

        if hasattr(persyn_config.memory, 'neo4j'):
            neomodel_config.DATABASE_URL = persyn_config.memory.neo4j.url

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
        Save a line of conversation to Elasticsearch. Returns the convo doc.
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
        Save a conversation summary to Elasticsearch.
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
        if not convo_id:
            return []

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

    def get_summary_by_id(self, convo_id):
        '''
        Fetch an conversation summary its convo_id.
        Returns the summary object.
        '''
        if not convo_id:
            return []

        ret = self.es.search( # pylint: disable=unexpected-keyword-arg
            index=self.index['summary'],
            query={
                "term": {"convo_id.keyword": convo_id}
            },
            sort=[{"@timestamp":{"order":"asc"}}],
            size=1
        )['hits']['hits']

        log.debug(f"get_summary_by_id({convo_id}):", ret)
        return ret

    def lookup_summaries(self, service, channel, search=None, size=3):
        '''
        Return a list of summaries matching the search term for this channel.
        '''
        # TODO: match speaker id HERE when cross-channel entity merging is working
        query = {
            "bool": {
                "should": [
                    {
                        "bool": {
                            "must": [
                                {"match": {"service.keyword": "import_service"}},
                            ]
                        }
                    },
                    {
                        "bool": {
                            "must": [
                                {"match": {"service.keyword": service}},
                                {"match": {"channel.keyword": channel}}
                            ],
                        }
                    }
                ],
            }
        }

        if search:
            for i in range(len(query['bool']['should'])):
                query['bool']['should'][i]['bool']['must'].append({"match": {"convo": {"query": search}}})

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
                "should": [
                    {
                        "bool": {
                            "must": [
                                {"match": {"service.keyword": "import_service"}},
                            ]
                        }
                    },
                    {
                        "bool": {
                            "must": [
                                {"match": {"service.keyword": service}},
                                {"match": {"channel.keyword": channel}}
                            ],
                        }
                    }
                ],
            }
        }

        if search:
            for i in range(len(query['bool']['should'])):
                query['bool']['should'][i]['bool']['must'].append({"match": {"convo": {"query": search}}})

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
            'graph': self.relationships.graph_to_json(
                self.relationships.get_relationship_graph(text, include_archetypes=include_archetypes)
            ),
            'convo': text,
            'refresh': False
        }
        rep = self.save_relationship(**doc)
        if rep['result'] != 'created':
            log.critical("📉 Could not save relationship:", rep)
        return rep['result']

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

    def name_to_entity_id(self, service, channel, name):
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
            entity_id = self.name_to_entity_id(service, channel, speaker_id)
        entity = self.lookup_entity_id(entity_id)

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

    def lookup_entity_id(self, entity_id):
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

    def lookup_speaker_name(self, speaker_name, size=1000):
        '''
        Look up a speaker by name in the Elasticsearch entities index.
        It returns all matching entities regardless of service/channel.
        '''
        hits = self.es.search(  # pylint: disable=unexpected-keyword-arg
            index=self.index['entity'],
            query={
                "term": {"speaker_name.keyword": speaker_name}
            },
            size=size
        )['hits']['hits']

        ret = []

        for hit in hits:
            ret.append(hit['_source'])

        return ret

    def entity_id_to_name(self, entity_id):
        ''' If the entity_id exists, return its name. '''
        entity = self.lookup_entity_id(entity_id)
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

    def find_related_convos(self, service, channel, convo, size=1, edge_bias=0.5):
        '''
        Find conversations related to convo using ES score and graph analysis.

        Returns a ranked list of graph hits.
        '''
        if not convo:
            return []

        convo_text = ' '.join(convo)

        # No relationships? Nothing to match.
        hits = self.lookup_relationships(service, channel, convo_text, size)
        if not hits:
            log.info("👨‍👩‍👧‍👦 find_related_convos():", "No hits, nothing to match.")
            return []

        G = self.relationships.get_relationship_graph(convo_text)

        ranked = self.relationships.ranked_matches(G, hits, edge_bias=edge_bias)
        log.info("👨‍👩‍👧‍👦 find_related_convos():", f"{len(ranked)} matches")
        return ranked

    def add_goal(self, service, channel, goal, refresh=True):
        '''
        Add a goal to a channel. Returns the top 10 unachieved goals.
        '''
        cur_ts = get_cur_ts()
        doc = {
            "service": service,
            "channel": channel,
            "@timestamp": cur_ts,
            "goal": goal,
            "achieved": False
        }
        _id = self.es.index(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            index=self.index['goal'],
            document=doc,
            refresh='true' if refresh else 'false'
        )["_id"]

        log.debug("doc:", _id)

        return self.get_goals(service, channel, achieved=False)

    def achieve_goal(self, service, channel, goal):
        '''
        Set a goal to the achieved state. Returns the top ten unachieved goals.
        '''
        for doc in self.get_goals(service, channel, goal, achieved=False):
            doc['_source']['achieved'] = True
            doc['_source']['achieved_on'] = get_cur_ts()
            self.es.update(index=self.index['goal'], id=doc['_id'], doc=doc['_source'], refresh=True)

        return self.get_goals(service, channel, achieved=False)

    def get_goals(self, service, channel, goal=None, achieved=None, size=10):
        '''
        Return goals for a channel. Returns the 10 most recent goals by default.
        Set achieved to True or False to return only achieved or unachieved goals.
        Specify a goal to return only that specific goal.
        '''
        ret = []

        query = {
            "bool": {
                "must": [
                    {"match": {"service.keyword": service}},
                    {"match": {"channel.keyword": channel}},
                ]
            }
        }
        if goal:
            query["bool"]["must"].append({"match": {"goal": goal}})
        if achieved is not None:
            query["bool"]["must"].append({"match": {"achieved": achieved}})

        ret = self.es.search(  # pylint: disable=unexpected-keyword-arg
            index=self.index['goal'],
            query=query,
            sort=[{"@timestamp": {"order": "desc"}}],
            size=size
        )['hits']['hits']

        log.debug(f"🥇 return: {ret}")
        return ret

    def list_goals(self, service, channel, achieved=False, size=10):
        '''
        Return a simple list of goals for a channel. Returns the 10 most recent goals by default.
        '''
        ret = []
        for goal in self.get_goals(service, channel, goal=None, achieved=achieved, size=size):
            ret.append(goal['_source']['goal'])
        return ret

    def add_news(self, service, channel, url, title, refresh=True):
        '''
        Add a news url that we've read. Returns the doc _id.
        '''
        cur_ts = get_cur_ts()
        doc = {
            "service": service,
            "channel": channel,
            "@timestamp": cur_ts,
            "url": url,
            "title": title
        }
        _id = self.es.index(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            index=self.index['news'],
            document=doc,
            refresh='true' if refresh else 'false'
        )["_id"]

        log.debug("🗞️ doc:", _id)

        return _id

    def have_read(self, service, channel, url):
        '''
        Return True if we have read this article, otherwise False.
        '''
        ret = []

        query = {
            "bool": {
                "must": [
                    {"match": {"service.keyword": service}},
                    {"match": {"channel.keyword": channel}},
                    {"match": {"url.keyword": url}}
                ]
            }
        }

        ret = bool(self.es.search(  # pylint: disable=unexpected-keyword-arg
            index=self.index['news'],
            query=query,
            sort=[{"@timestamp": {"order": "desc"}}],
            size=1
        )['hits']['hits'])

        log.debug(f"🗞️ {url}: {ret}")
        return ret

    def delete_all_nodes(self, confirm=False):
        ''' Delete all graph nodes for this bot '''
        if confirm is not True:
            return False

        for node in Entity.nodes.filter(Q(bot_id=self.bot_id)):
            node.delete()

        return True

    def fetch_all_nodes(self, node_type=None):
        ''' Return all graph nodes for this bot '''
        if node_type is None:
            return Entity.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'person':
            return Person.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'thing':
            return Thing.nodes.filter(Q(bot_id=self.bot_id))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    def find_node(self, name, node_type=None):
        ''' Return all nodes with the given name'''
        if node_type is None:
            return Entity.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'person':
            return Person.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'thing':
            return Thing.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    @staticmethod
    def safe_name(name): # TODO: unify this with gpt.py
        ''' Return name sanitized as alphanumeric, space, or comma only, max 64 characters. '''
        return re.sub(r"[^a-zA-Z0-9, ]+", '', name.strip())[:64]

    def shortest_path(self, src, dest, src_type=None, dest_type=None):
        '''
        Find the shortest path between two nodes, if any.
        If src_type or dest_type are specified, constrain the search to nodes of that type.
        Returns a list of triples (string names of nodes and edges) encountered along the path.
        '''
        safe_src = self.safe_name(src)
        safe_dest = self.safe_name(dest)

        if safe_src == safe_dest:
            return []

        query = f"""
        MATCH
        (a{':'+src_type if src_type else ''} {{name: '{safe_src}', bot_id: '{self.bot_id}'}}),
        (b{':'+dest_type if dest_type else ''} {{name: '{safe_dest}', bot_id: '{self.bot_id}'}}),
        p = shortestPath((a)-[*]-(b))
        WITH p
        WHERE length(p) > 1
        RETURN p
        """
        paths = neomodel_db.cypher_query(query)[0]
        ret = []
        if not paths:
            return ret

        for r in paths[0][0].relationships:
            ret.append((r.start_node.get('name'), r.get('verb'), r.end_node.get('name')))

        return ret

    def triples_to_kg(self, triples):
        '''
        Convert subject, predicate, object triples into a Neo4j graph.
        '''
        if not hasattr(self.persyn_config.memory, 'neo4j'):
            log.error('፨ No graph server defined, cannot call triples_to_kg()')
            return

        speaker_names = set()
        thing_names = set()
        speakers = {}

        for triple in triples:
            (s, _, o) = triple

            if s not in thing_names and self.lookup_speaker_name(s):
                speaker_names.add(s)
            else:
                thing_names.add(s)

            if o not in thing_names and self.lookup_speaker_name(o):
                speaker_names.add(o)
            else:
                thing_names.add(o)

        with neomodel_db.transaction:
            for name in speaker_names:
                try:
                    speakers[name] = Person.nodes.get(name=name, bot_id=self.bot_id)  # pylint: disable=no-member
                # If they don't yet exist in the graph, make a new node
                except Person.DoesNotExist:  # pylint: disable=no-member
                    speakers[name] = Person(name=name, bot_id=self.bot_id).save()

            things = {}
            for t in Thing.get_or_create(*[{'name': n, 'bot_id': self.bot_id} for n in list(thing_names) if n not in speakers]):
                things[t.name] = t

            for link in triples:
                if link[0] in speakers:
                    subj = speakers[link[0]]
                else:
                    subj = things[link[0]]

                pred = link[1]

                if link[2] in speakers:
                    obj = speakers[link[2]]
                else:
                    obj = things[link[2]]

                found = False
                for rel in subj.link.all_relationships(obj):
                    if rel.verb == pred:
                        found = True
                        rel.weight = rel.weight + 1
                        rel.save()

                if not found:
                    rel = subj.link.connect(obj, {'verb': pred})
