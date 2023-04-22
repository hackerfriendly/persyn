''' memory.py: long and short term memory by Elasticsearch. '''
# pylint: disable=invalid-name, no-name-in-module, abstract-method, no-member
import re
import uuid
import logging
from datetime import datetime
import os

from abc import ABC
from typing import Optional
from urllib.parse import urlparse

import ulid

import redis

from redis_om import (
    Field,
    HashModel,
    Migrator
)

# Time
from persyn.interaction.chrono import elapsed, get_cur_ts

# Relationship graphs
from persyn.interaction.relationships import Relationships

# Color logging
from persyn.utils.color_logging import ColorLog

log = ColorLog()


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

        if convo_id is None:
            convo_id = str(ulid.ULID())

        return self.stm.append(
            service,
            channel,
            self.ltm.save_convo(
                service=service,
                channel=channel,
                speaker_name=speaker_name,
                msg=msg,
                convo_id=convo_id,
                speaker_id=speaker_id,
                verb=verb
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
                f"{line.speaker_name}: {line.msg}" for line in convo
                if 'verb' not in line or line.verb == 'dialog'
            ]
        return []

    def convo(self, service, channel, feels=False):
        '''
        Return the entire convo from stm (if any).

        Result is a list of "speaker: msg" or "speaker (verb): msg" strings.
        '''
        convo = self.stm.fetch(service, channel)
        if not convo:
            return []

        ret = []
        for msg in convo:
            if msg.verb in ['dialog', None]:
                ret.append(f"{msg.speaker_name}: {msg.msg}")
            elif feels or msg.verb != 'feels':
                ret.append(f"{msg.speaker_name} {msg.verb}: {msg.msg}")

        return ret

    def feels(self, convo_id):
        '''
        Return the last known feels for this channel.
        '''
        convo = self.ltm.get_convo_by_id(convo_id)
        log.debug("📢", convo)
        for doc in convo[::-1]:
            if doc.verb == 'feels':
                return doc.msg

        return "nothing in particular"

    def summaries(self, service, channel, size=3):
        ''' Return the summary text from ltm (if any) '''
        return [s.summary for s in self.ltm.lookup_summaries(service, channel, None, size=size)]

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
        self.convo[service][channel]['id'] = str(ulid.ULID())
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




class BaseModel(HashModel, ABC):
    class Meta:
        global_key_prefix = "persyn"
        database = redis.Redis().from_url(os.environ['REDIS_OM_URL'])
        index_name = "MyCustomIndex"
        _ = Migrator().run()

# 'persyn':bot_id:'convo':pk
class Convo(BaseModel):
    service: str = Field(index=True)
    channel: str = Field(index=True)
    convo_id: str = Field(index=True)
    speaker_name: str = Field(index=True)
    speaker_id: str = Field(index=True)
    msg: str = Field(index=True, full_text_search=True)
    verb: str = Field(index=True)

# 'persyn':bot_id:'summary':pk
class Summary(BaseModel):
    service: str = Field(index=True)
    channel: str = Field(index=True)
    convo_id: str = Field(index=True)
    summary: str = Field(index=True, full_text_search=True)
    keywords: Optional[str] = Field(index=True)

# LTM object
class LongTermMemory(): # pylint: disable=too-many-arguments
    ''' Wrapper class for Elasticsearch conversational memory. '''
    def __init__(self, persyn_config, version=None):
        self.relationships = Relationships(persyn_config)
        self.persyn_config = persyn_config
        self.bot_name = persyn_config.id.name
        self.bot_id = uuid.UUID(persyn_config.id.guid)
        self.bot_ulid = ulid.ULID().from_uuid(self.bot_id)

        self.redis = redis.from_url(persyn_config.memory.redis)

    def shortest_path(self, src, dest, src_type=None, dest_type=None):
        ''' TODO: REMOVE THIS STUB AND REPLACE WITH graph.py '''
        return []

    def get_last_timestamp(self, service, channel):
        '''
        Get the timestamp of the last message, or the current ts if there is none.
        '''
        msg = self.get_last_message(service, channel)
        if msg:
            return get_cur_ts(epoch=ulid.ULID().from_str(msg.pk).timestamp)
        else:
            return get_cur_ts()


    def save_convo(
        self,
        service,
        channel,
        speaker_name,
        msg,
        convo_id=ulid.ULID(),
        speaker_id=None,
        verb='dialog'
    ):
        '''
        Save a line of conversation. Returns the Convo object.
        '''
        # Save speaker entity
        if speaker_id is None:
            # speaker_id = self.get_speaker_id(service, channel, speaker)
            speaker_id = "unknown speaker"

        convo_line = Convo(
            service=service,
            channel=channel,
            convo_id=convo_id,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            msg=msg,
            verb=verb
        )
        convo_line.Meta.model_key_prefix = f"{self.bot_id}:convo"

        return convo_line.save()

    # TODO: set refresh=False after separate summary thread is implemented.
    def save_summary(self, service, channel, convo_id, summary, keywords=None):
        '''
        Save a conversation summary to memory. Returns the Summary object.
        '''
        the_summary = Summary(
            service=service,
            channel=channel,
            convo_id=convo_id,
            summary=summary,
            keywords=' '.join(keywords)
        )
        the_summary.Meta.model_key_prefix = f"{self.bot_id}:summary"

        return the_summary.save()

    def get_last_message(self, service, channel):
        ''' Return the last message seen on this channel '''
        try:
            return Convo.find((Convo.service == service) & (Convo.channel == channel)).all()[-1]
        except IndexError:
            return None

    def get_convo_by_id(self, convo_id):
        ''' Return all Convo objects matching convo_id in chronological order '''
        return Convo.find(Convo.convo_id == convo_id).all()

    def get_summary_by_id(self, convo_id):
        ''' Return the last summary for this convo_id '''
        try:
            return Summary.find(Summary.convo_id == convo_id).all()[-1]
        except IndexError:
            return None

    def lookup_summaries(self, service, channel, search=None, size=None):
        '''
        Return a list of summaries matching the search term for this channel.
        '''
        # log.warning("Running Migrator")
        # Migrator().run()

        log.warning(f"lookup_summaries(): {service} {channel} {search} {size}")
        if search is None:
            log.warning("find with no search term to server:", os.environ['REDIS_OM_URL'])
            return Summary.find((Summary.service == service) & (Summary.channel == channel)).all()[:size]

        # Do we need to escape search terms?
        search = search.replace('%', '')
        return Summary.find((Summary.service == service) & (Summary.channel == channel) & (Summary.summary % search)).all()[:size]


    def lookup_relationships(self, service, channel, search=None, size=3):
        '''
        Return a list of convo graphs matching the search term for this channel.
        '''
        return []
        # # TODO: match speaker id HERE when cross-channel entity merging is working
        # query = {
        #     "bool": {
        #         "should": [
        #             {
        #                 "bool": {
        #                     "must": [
        #                         {"match": {"service.keyword": "import_service"}},
        #                     ]
        #                 }
        #             },
        #             {
        #                 "bool": {
        #                     "must": [
        #                         {"match": {"service.keyword": service}},
        #                         {"match": {"channel.keyword": channel}}
        #                     ],
        #                 }
        #             }
        #         ],
        #     }
        # }

        # if search:
        #     for i in range(len(query['bool']['should'])):
        #         query['bool']['should'][i]['bool']['must'].append({"match": {"convo": {"query": search}}})

        # history = self.es.search( # pylint: disable=unexpected-keyword-arg
        #     index=self.index['relationship'],
        #     query=query,
        #     size=size
        # )['hits']['hits']

        # ret = []
        # for hit in history[::-1]:
        #     ret.append(hit)

        # log.debug(f"lookup_relationships(): {ret}")
        # return ret

    def save_relationship_graph(self, service, channel, convo_id, text, include_archetypes=True):
        ''' Save a relationship graph '''
        return True
        # doc = {
        #     '@timestamp': get_cur_ts(),
        #     'service': service,
        #     'channel': channel,
        #     'convo_id': convo_id,
        #     'graph': self.relationships.graph_to_json(
        #         self.relationships.get_relationship_graph(text, include_archetypes=include_archetypes)
        #     ),
        #     'convo': text,
        #     'refresh': False
        # }
        # rep = self.save_relationship(**doc)
        # if rep['result'] != 'created':
        #     log.critical("📉 Could not save relationship:", rep)
        # return rep['result']

    @staticmethod
    def entity_key(service, channel, name):
        ''' Unique string for each service + channel + name '''
        return f"{str(service).strip()}|{str(channel).strip()}|{str(name).strip()}"

    @staticmethod
    def uuid_to_entity(the_uuid):
        ''' Return the equivalent short ID (str) for a uuid (str) '''
        return str(ulid.ULID().from_uuid(the_uuid))

    @staticmethod
    def entity_to_uuid(entity_id):
        ''' Return the equivalent UUID (str) for a short ID (str) '''
        return ulid.ULID().from_str(str(entity_id)).to_uuid()

    def name_to_entity_id(self, service, channel, name):
        ''' One distinct short UUID per bot_id + service + channel + name '''
        return self.uuid_to_entity(uuid.uuid5(self.bot_id, self.entity_key(service, channel, name)))

    def lookup_speaker_name(self, speaker_name, size=1000):
        return None
        # '''
        # Look up a speaker by name in the Elasticsearch entities index.
        # It returns all matching entities regardless of service/channel.
        # '''
        # hits = self.es.search(  # pylint: disable=unexpected-keyword-arg
        #     index=self.index['entity'],
        #     query={
        #         "term": {"speaker_name.keyword": speaker_name}
        #     },
        #     size=size
        # )['hits']['hits']

        # ret = []

        # for hit in hits:
        #     ret.append(hit['_source'])

        # return ret

    def save_opinion(self, service, channel, topic, opinion, speaker_id=None, refresh=True):
        '''
        Save an opinion to Elasticscarch.
        '''
        return None

        # if not speaker_id:
        #     speaker_id = self.bot_id

        # doc = {
        #     "service": service,
        #     "channel": channel,
        #     "topic": topic.lower(),
        #     "opinion": opinion,
        #     "speaker_id": speaker_id,
        #     "@timestamp": get_cur_ts()
        # }
        # self.es.index( # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        #     index=self.index['opinion'],
        #     document=doc,
        #     refresh='true' if refresh else 'false'
        # )
        # # return something here?

    def lookup_opinion(self, topic, service=None, channel=None, speaker_id=None, size=10):
        ''' Look up an opinion in Elasticsearch. '''
        return []
        # query = {
        #     "bool": {
        #         "must": [
        #             {"match": {"topic.keyword": topic.lower()}}
        #         ]
        #     }
        # }

        # if service:
        #     query["bool"]["must"].append({"match": {"service.keyword": service}})

        # if channel:
        #     query["bool"]["must"].append({"match": {"channel.keyword": channel}})

        # if speaker_id:
        #     query["bool"]["must"].append({"match": {"speaker_id.keyword": speaker_id}})

        # ret = []
        # for opinion in self.es.search( # pylint: disable=unexpected-keyword-arg
        #     index=self.index['opinion'],
        #     query=query,
        #     size=size
        # )['hits']['hits']:
        #     ret.append(opinion["_source"]["opinion"])

        # return ret

    def save_relationship(self, service, channel, refresh=True, **kwargs):
        '''
        Save a relationship to Elasticearch.
        '''
        return []

        # if 'source_id' in kwargs and not all(['rel' in kwargs, 'target_id' in kwargs]):
        #     log.critical('👩‍👧‍👦 source_id requires rel and target_id')
        #     return None

        # if 'convo_id' in kwargs and 'graph' not in kwargs:
        #     log.critical('👩‍👧‍👦 convo_id requires graph')
        #     return None

        # cur_ts = get_cur_ts()
        # doc = {
        #     "@timestamp": cur_ts,
        #     "service": service,
        #     "channel": channel
        # }

        # for term, val in kwargs.items():
        #     doc[term] = val

        # ret = self.es.index( # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        #     index=self.index['relationship'],
        #     document=doc,
        #     refresh='true' if refresh else 'false'
        # )
        # return ret

    def lookup_relationship(self, service, channel, size=10, **kwargs):
        ''' Look up a relationship in Elasticsearch. '''
        ret = []

        return ret

        # query = {
        #     "bool": {
        #         "must": [
        #             {"match": {"service.keyword": service}},
        #             {"match": {"channel.keyword": channel}}
        #         ]
        #     }
        # }

        # for term, val in kwargs.items():
        #     query['bool']['must'].append({"match": {f'{term}.keyword': val}})

        # log.debug(f"👨‍👩‍👧 query: {query}")

        # ret = self.es.search( # pylint: disable=unexpected-keyword-arg
        #     index=self.index['relationship'],
        #     query=query,
        #     sort=[{"@timestamp":{"order":"desc"}}],
        #     size=size
        # )['hits']['hits']

        # log.debug(f"👨‍👩‍👧 return: {ret}")
        # return ret

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
        return []
        # cur_ts = get_cur_ts()
        # doc = {
        #     "service": service,
        #     "channel": channel,
        #     "@timestamp": cur_ts,
        #     "goal": goal,
        #     "achieved": False
        # }
        # _id = self.es.index(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        #     index=self.index['goal'],
        #     document=doc,
        #     refresh='true' if refresh else 'false'
        # )["_id"]

        # log.debug("doc:", _id)

        # return self.get_goals(service, channel, achieved=False)

    def achieve_goal(self, service, channel, goal):
        '''
        Set a goal to the achieved state. Returns the top ten unachieved goals.
        '''
        return []
        # for doc in self.get_goals(service, channel, goal, achieved=False):
        #     doc['_source']['achieved'] = True
        #     doc['_source']['achieved_on'] = get_cur_ts()
        #     self.es.update(index=self.index['goal'], id=doc['_id'], doc=doc['_source'], refresh=True)

        # return self.get_goals(service, channel, achieved=False)

    def get_goals(self, service, channel, goal=None, achieved=None, size=10):
        '''
        Return goals for a channel. Returns the 10 most recent goals by default.
        Set achieved to True or False to return only achieved or unachieved goals.
        Specify a goal to return only that specific goal.
        '''
        ret = []
        return ret

        # query = {
        #     "bool": {
        #         "must": [
        #             {"match": {"service.keyword": service}},
        #             {"match": {"channel.keyword": channel}},
        #         ]
        #     }
        # }
        # if goal:
        #     query["bool"]["must"].append({"match": {"goal": goal}})
        # if achieved is not None:
        #     query["bool"]["must"].append({"match": {"achieved": achieved}})

        # ret = self.es.search(  # pylint: disable=unexpected-keyword-arg
        #     index=self.index['goal'],
        #     query=query,
        #     sort=[{"@timestamp": {"order": "desc"}}],
        #     size=size
        # )['hits']['hits']

        # log.debug(f"🥇 return: {ret}")
        # return ret

    def list_goals(self, service, channel, achieved=False, size=10):
        '''
        Return a simple list of goals for a channel. Returns the 10 most recent goals by default.
        '''
        ret = []
        for goal in self.get_goals(service, channel, goal=None, achieved=achieved, size=size):
            ret.append(goal.goal)
        return ret

    def add_news(self, service, channel, url, title, refresh=True):
        '''
        Add a news url that we've read. Returns the doc _id.
        '''
        return None
        # cur_ts = get_cur_ts()
        # doc = {
        #     "service": service,
        #     "channel": channel,
        #     "@timestamp": cur_ts,
        #     "url": url,
        #     "title": title
        # }
        # _id = self.es.index(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        #     index=self.index['news'],
        #     document=doc,
        #     refresh='true' if refresh else 'false'
        # )["_id"]

        # log.debug("🗞️ doc:", _id)

        # return _id

    def have_read(self, service, channel, url):
        '''
        Return True if we have read this article, otherwise False.
        '''
        ret = []
        return False

        # query = {
        #     "bool": {
        #         "must": [
        #             {"match": {"service.keyword": service}},
        #             {"match": {"channel.keyword": channel}},
        #             {"match": {"url.keyword": url}}
        #         ]
        #     }
        # }

        # ret = bool(self.es.search(  # pylint: disable=unexpected-keyword-arg
        #     index=self.index['news'],
        #     query=query,
        #     sort=[{"@timestamp": {"order": "desc"}}],
        #     size=1
        # )['hits']['hits'])

        # log.debug(f"🗞️ {url}: {ret}")
        # return ret
